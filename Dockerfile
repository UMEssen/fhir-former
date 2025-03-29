FROM python:3.9 as poetry2requirements
COPY pyproject.toml poetry.lock /
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.4.2
ENV PATH="/opt/poetry/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /opt/poetry/bin && \
    ln -s poetry poetry1.4
RUN poetry1.4 export --without-hashes --format requirements.txt \
    | grep -v "torch=" \
    > /Requirements.txt


FROM nvcr.io/nvidia/pytorch:23.08-py3

# Install app dependencies
COPY --from=poetry2requirements /Requirements.txt /tmp
RUN pip3 install -U pip && \
    pip3 install -r /tmp/Requirements.txt && \
    rm /tmp/Requirements.txt

WORKDIR /app

ENV TRANSFORMERS_CACHE=/tmp/.cache/transformers
ENV HF_DATASETS_CACHE=/tmp/.cache/huggingface/datasets
ENV HF_HOME=/tmp/.cache/huggingface

CMD ["tail", "-f", "/dev/null"]
