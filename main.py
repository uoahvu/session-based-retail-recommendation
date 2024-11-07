from data_preprocess import preprocess

if __name__ == "__main__":
    df = preprocess()
    print(df)

# worthy_visitors = session_category_events.groupby('visitorid').size().sort_values()[6000:].index.tolist()
# data_sample = session_category_events[session_category_events['visitorid'].isin(worthy_visitors)]

# data_sample = data_sample.groupby(['visitorid', 'session']).filter(lambda x: len(x) >= 5)

# item_list = np.unique(data_sample["itemid"].values)
# item2idx = {item: idx for idx, item in enumerate(item_list)}
# def item_id2idx(x):
#     return item2idx[x]
# data_sample['itemidx'] = data_sample['itemid'].apply(lambda x : item_id2idx(x))

# # 사용자로 나누기
# visitor_list = data_sample['visitorid'].unique().tolist()
# train_ratio = 0.8
# train_users = visitor_list[:int(len(visitor_list)*train_ratio)]
# test_users = visitor_list[int(len(visitor_list)*train_ratio):]

# train_data_sample = data_sample[data_sample['visitorid'].isin(train_users)]
# test_data_sample = data_sample[data_sample['visitorid'].isin(test_users)]

# n_items = data_sample['itemidx'].nunique()

# train_dataset_sample = SessionDataset(train_data_sample, n_items)
# test_dataset_sample = SessionDataset(test_data_sample, n_items)
# train_loader_sample = DataLoader(train_dataset_sample, batch_size=250, shuffle=True)
# test_loader_sample = DataLoader(test_dataset_sample, batch_size=250, shuffle=False)

# # Train the model
# num_epochs = 1000
# total_loss = 0
# for epoch in range(num_epochs):
#     start_time = time.time()
#     for i, (inputs, target) in enumerate(train_loader_sample): # 사용자별 데이터 개수 다름
#         print(inputs)
#         print(len(inputs))
#         print(target)
#         print(len(target))
