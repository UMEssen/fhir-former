import json
from pathlib import Path
from typing import Dict, Union

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PatientEncounterDataset(Dataset):
    def __init__(self, config, max_length=None, num_samples=None):
        self.config = config

        model_path = Path(self.config["model_checkpoint"])
        if model_path.exists() and not (model_path / "tokenizer.json").exists():
            with (model_path / "config.json").open("rb") as f:
                model_name = json.load(f)["_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(str(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_checkpoint"])
        with open(self.config["task_dir"] / "train.json", "r") as f:
            self.data = json.load(f)[:num_samples] if num_samples else json.load(f)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def prepare_used_items(self, idx: int) -> Dict[str, Union[int, str]]:
        encoding = self.tokenizer(
            self.data[idx]["text"],
            truncation=self.config["truncation"],
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "patient_id": self.data[idx]["patient_id"],
            "encounter_id": self.data[idx]["encounter_id"],
            "text": self.data[idx]["text"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def __getitem__(self, idx):
        raise NotImplementedError
