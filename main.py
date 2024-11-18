from data_preprocess import preprocess, data_augmentation
from data_loader import SessionDataset
from session_based_recommender import SessionLSTM

import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    df, candidate_dict = preprocess()

    # 유저 기준 Train/Test Split
    print("Train/Test Spliting ...")
    visitor_list = df["visitorid"].unique().tolist()
    train_ratio = 0.8
    train_users = visitor_list[: int(len(visitor_list) * train_ratio)]
    test_users = visitor_list[int(len(visitor_list) * train_ratio) :]

    train_data_sample = df[df["visitorid"].isin(train_users)]
    # print("Train Data Augmentation ...")
    # train_data_sample = data_augmentation(train_data_sample, 2)
    test_data_sample = df[df["visitorid"].isin(test_users)]

    n_items = df["itemidx"].nunique()
    n_categorys = df["categoryidx"].nunique()
    n_pcategorys = df["pcategoryidx"].nunique()

    train_dataset_sample = SessionDataset(train_data_sample, n_items)
    test_dataset_sample = SessionDataset(test_data_sample, n_items)
    train_loader_sample = DataLoader(train_dataset_sample, batch_size=250, shuffle=True)
    test_loader_sample = DataLoader(test_dataset_sample, batch_size=250, shuffle=False)

    print("Session LSTM Training ...")
    num_epochs = 100
    hidden_size = 128
    model = SessionLSTM(
        num_epochs,
        n_items,
        n_categorys,
        n_pcategorys,
        hidden_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train_model(
        train_loader_sample, test_loader_sample, optimizer, candidate_dict
    )
