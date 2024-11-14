import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, data, num_items):
        self.data = data
        self.session_data = data.groupby(["visitorid", "session"])
        self.num_items = num_items
        self.seq_len = 100

    def __len__(self):
        return len(self.session_data)

    def __getitem__(self, idx):
        session_key = list(self.session_data.groups.keys())[idx]
        session_data = self.session_data.get_group(session_key)

        X = (
            session_data[session_data["event"] == "view"]
            .sort_values("timestamp")[["itemidx", "categoryidx", "pcategoryidx"]]
            .values
        )
        y = (
            session_data[session_data["event"] != "view"]
            .sort_values("timestamp")["itemidx"]
            .values
        )

        X_new = torch.zeros(self.seq_len, len(X[0]))

        if len(X) > self.seq_len:
            X_new[:, :] = torch.tensor(X[: self.seq_len])
        else:
            X_new[: len(X), :] = torch.tensor(X)

        return X_new, torch.zeros(self.num_items, dtype=torch.float32).scatter_(
            0, torch.tensor(y), torch.ones(len(y), dtype=torch.float32)
        )
