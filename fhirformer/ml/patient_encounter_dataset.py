import json
from typing import Dict, Union

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PatientEncounterDataset(Dataset):
    def __init__(self, config, max_length=None, num_samples=None):
        self.config = config
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
