#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import gc
import hashlib
import logging
import multiprocessing as mp
import os
import random
import re
import struct
import sys
import time
from ast import List, Tuple
from collections import defaultdict
from itertools import tee
from logging import Logger
from pathlib import Path
from typing import Any, Iterable

import datasets
import numpy as np
import psutil
from datasets import load_dataset
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.integrate import quad as integrate
from tqdm import tqdm

SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()


def ngrams(sequence: list[str], n: int, min_ngram_size: int) -> Iterable:
    """
    Directly taken from nltk package to avoid dependency.

    Parameters
    ----------
    sequence : list
        The sequence of items to be n-grammed.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.
    """
    if len(sequence) < min_ngram_size:
        return []
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: list[tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
) -> dict[str, Any]:
    """
    Combined with some datasketch code to better parallelize computation.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.
    """
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }
    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens],
        dtype=np.uint64,
    )  # noqa: E501
    a, b = permutations
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME,
        MAX_HASH,
    )  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]

    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


def run_deduplication(
    input_dir: Path,  # noqa: E501
    column: str,
    cache_dir: str,
    ngram_size: int,
    num_perm: int,
    threshold: float,
    min_ngram_size: int,
    output_file: Path,
    split: str,
):
    global uf
    logging.basicConfig(level=logging.INFO)

    time_measures = {}
    start_time = time.time()

    bands, rows = optimal_param(threshold, num_perm)

    HASH_RANGES = [(i * rows, (i + 1) * rows) for i in range(bands)]
    HASH_TABLES: list = [defaultdict(set) for _ in range(bands)]

    time_measures["load_dataset"] = time.time()
    ds = load_dataset(
        str(input_dir),
        split=split,
        cache_dir=cache_dir
        # num_proc=os.cpu_count(),
    )
    # ds = pl.scan_ndjson(input_dir)
    time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]
    DATA_SIZE = len(ds)
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T

    time_measures["minhash"] = time.time()
    embedded = ds.map(
        function=embed_func,
        fn_kwargs={
            "num_perm": num_perm,
            "hashranges": HASH_RANGES,
            "ngram_size": ngram_size,
            "permutations": PERMUTATIONS,
            "min_ngram_size": min_ngram_size,
        },
        input_columns=[column],
        remove_columns=ds.column_names,
        num_proc=os.cpu_count(),
        with_indices=True,
        desc="Fingerprinting...",
    )
    time_measures["minhash"] = time.time() - time_measures["minhash"]

    time_measures["clustering"] = time.time()
    batch_size: int = 10000
    for i in tqdm(
        range(0, len(embedded), batch_size),
        dynamic_ncols=True,
        desc="Iterating MinHashes...",  # noqa: E501
    ):
        batch = embedded[i : i + batch_size]
        for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
            for H, hashtable in zip(Hs, HASH_TABLES):
                hashtable[H].add(key)
    for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                uf.union(x, idx)
    time_measures["clustering"] = time.time() - time_measures["clustering"]

    time_measures["filtering"] = time.time()
    gc.freeze()
    gc.disable()
    ds = ds.map(
        function=lambda _, idx: {"__cluster__": uf.find(idx)},
        with_indices=True,
        num_proc=os.cpu_count(),
        new_fingerprint=str(random.getrandbits(128)),
        desc="Finding clusters...",
    )
    gc.enable()
    gc.collect()
    # This is where the deduplication happens
    # Since there is no easy groupby in datasets
    # I will use this simple filter for now
    final_data = ds.filter(
        function=lambda record, idx: record["__cluster__"] == idx,
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Filtering clusters...",
    )
    time_measures["filtering"] = time.time() - time_measures["filtering"]

    time_measures["save"] = time.time()
    final_data = final_data.remove_columns(["__cluster__"])
    final_data.to_json(output_file)
    time_measures["save"] = time.time() - time_measures["save"]

    FINAL_DATA_SIZE = len(final_data)
    DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
    PAD = 32

    for key, value in time_measures.items():
        logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
    logger.info(f"{'Data Number (before)':<{PAD}}: {DATA_SIZE}")
    logger.info(
        f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / DATA_SIZE:.2%})",  # noqa: E501
    )
    logger.info(
        f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / DATA_SIZE:.2%})",
    )  # noqa: E501
    logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
    logger.info(f"{'Deduplicated Dataset':<{PAD}}: {output_file}")
    logger.info("🤗 Happy Deduplicating 🤗")


mp.set_start_method("fork", force=True)
uf = UnionFind()


# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]


def generate_hash_values(
    content: str,
    idx: int,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int,
) -> List[Tuple[int, bytes, int]]:
    """
    Generate the MinHashLSH values for a given document.

    Parameters
    ----------
    content : str
        The content of the document.
    idx : int
        The index of the document.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of the n-grams.
    hashranges : list
        The ranges of offsets for each hash value.
    permutations : np.ndarray
        The permutations for the hash values.
    min_ngram_size : int
        The minimum number of items in the sequence to generate n-grams.

    Returns
    -------
    List[Tuple[int, bytes, int]]
        The list of (band_idx, hash value, idx) for the document.
    """
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }
    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens],
        dtype=np.uint64,
    )
    a, b = permutations
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME,
        MAX_HASH,
    )
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges, we only need all nodes align to a single node, since
    we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


def run_deduplication_spark(
    input_dir: str,
    threshold: float,
    min_ngram_size: int,
    ngram_size: int,
    num_perm: int,
    column: str,
    output_dir: str,
):
    # for file_name in [file for file in os.listdir(input_dir) if file.endswith('.jsonl')]:
    #     with open(input_dir + file_name) as json_file:
    #         data = json.load(json_file)

    print(psutil.version_info)
    conf = SparkConf()
    conf.set("spark.app.name", "MinHashLSH")
    conf.set("spark.debug.maxToStringFields", "100")
    conf.set("spark.driver.bindAddress", "10.116.60.188")
    conf.set("spark.driver.memory", "15g")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(
        __name__
    )  # type: ignore

    B, R = optimal_param(threshold, num_perm)
    log.info(f"Using optimal parameters: {B=}, {R=}")

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T

    df = spark.read.json(input_dir)
    df = df.withColumn("__id__", F.monotonically_increasing_id()).cache()
    records = df.select("__id__", column).rdd
    records = records.repartition(num_perm * 2).cache()

    edges = (
        records.flatMap(
            lambda x: generate_hash_values(
                content=x[1],
                idx=x[0],
                num_perm=num_perm,
                ngram_size=ngram_size,
                hashranges=HASH_RANGES,
                permutations=PERMUTATIONS,
                min_ngram_size=min_ngram_size,
            ),
        )
        .groupBy(lambda x: (x[0], x[1]))
        .flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
        .distinct()
        .cache()
    )

    a = edges
    while True:
        b = (
            a.flatMap(large_star_map)
            .groupByKey()
            .flatMap(large_star_reduce)
            .distinct()
            .cache()
        )
        a = (
            b.map(small_star_map)
            .groupByKey()
            .flatMap(small_star_reduce)
            .distinct()
            .cache()
        )
        changes = a.subtract(b).union(b.subtract(a)).collect()
        if len(changes) == 0:
            break

    results = a.collect()
    if len(results) == 0:
        log.info("No components found.")
        df.write.option("maxRecordsPerFile", 300_000).option(
            "intermediateFormat",
            "orc",
        ).parquet(output_dir, mode="overwrite")
        sys.exit(0)

    components = spark.createDataFrame(results, schema=["__id__", "component"]).sort(
        ["component", "__id__"],
    )
    components.show()
    df = df.join(components, on="__id__", how="left")
    df = df.filter(F.col("component").isNull()).drop("__id__", "component").cache()
    df.write.option("maxRecordsPerFile", 1_000_000_000).json(
        output_dir, mode="overwrite"
    )


# run_deduplication(
#     # input_dir: str,  # noqa: E501
#     input_dir = 'wikipedia_med_de',
#     column = 'text',
#     # column: str,
#     cache_dir= 'wikipedia_med_de/tmp',
#     ngram_size=13,
#     num_perm=128,
#     threshold=0.85,
#     min_ngram_size=3,
#     output_dir='wikipedia_med_de/wikipedia_med_de_dedup',
#     split='train',
# )
