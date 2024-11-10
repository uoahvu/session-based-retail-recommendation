import numpy as np
from data_preprocess import preprocess
from data_loader import SessionDataset
from session_based_recommender import SessionLSTM
from torch.utils.data import DataLoader
import time


if __name__ == "__main__":
    df = preprocess()

    # 유저 기준 Train/Test Split
    print("Train/Test Spliting ...")
    visitor_list = df["visitorid"].unique().tolist()
    train_ratio = 0.8
    train_users = visitor_list[: int(len(visitor_list) * train_ratio)]
    test_users = visitor_list[int(len(visitor_list) * train_ratio) :]

    train_data_sample = df[df["visitorid"].isin(train_users)]
    test_data_sample = df[df["visitorid"].isin(test_users)]

    n_items = df["itemidx"].nunique()
    n_categorys = df["category"].nunique()
    n_pcategorys = df["parentid"].nunique()

    train_dataset_sample = SessionDataset(train_data_sample, n_items)
    test_dataset_sample = SessionDataset(test_data_sample, n_items)
    train_loader_sample = DataLoader(train_dataset_sample, batch_size=250, shuffle=True)
    test_loader_sample = DataLoader(test_dataset_sample, batch_size=250, shuffle=False)

    print("Session LSTM Training...")
    auc = SessionLSTM(
        n_items, n_categorys, n_pcategorys, train_loader_sample, test_loader_sample
    )