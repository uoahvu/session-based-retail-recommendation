import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm


def timestamp2datetime(x):
    if x > 10**10:
        x /= 1000  # Convert milliseconds to seconds
    return datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def session_flag(group):
    flag = 0
    flags = []
    transaction_event = False
    for event in group["event"]:
        if event == "transaction":
            transaction_event = True
        elif transaction_event & (event != "transaction"):
            transaction_event = False
            flag += 1
        flags.append(flag)
    group["session"] = flags
    return group


def preprocess():
    print("Data Reading ...")
    category_tree = pd.read_csv("./datasets/category_tree.csv")
    events = pd.read_csv("./datasets/events.csv")
    item_properties_part1 = pd.read_csv("./datasets/item_properties_part1.csv")
    item_properties_part2 = pd.read_csv("./datasets/item_properties_part2.csv")

    print("Data Preprocessing ...")
    events["timestamp"] = events["timestamp"].apply(lambda x: timestamp2datetime(x))

    positive_visitor = events[
        (events["event"] == "addtocart") | (events["event"] == "transaction")
    ]["visitorid"].unique()
    filtered_events = events[events["visitorid"].isin(positive_visitor)].sort_values(
        "timestamp"
    )

    session_events = filtered_events.groupby("visitorid", group_keys=False).apply(
        session_flag
    )
    # 구매 event 가 있는 session만 대상으로 가져오기
    session_events = session_events.groupby(["visitorid", "session"]).filter(
        lambda x: x["event"].isin(["transaction"]).any()
    )

    item_category = pd.concat(
        [
            item_properties_part1[item_properties_part1["property"] == "categoryid"],
            item_properties_part2[item_properties_part2["property"] == "categoryid"],
        ]
    )
    item_category_dict = item_category.set_index("itemid")["value"].to_dict()

    def item_category_mapping(x):
        if x in item_category_dict:
            return int(item_category_dict[x])
        else:
            return -1

    session_events["category"] = session_events["itemid"].apply(
        lambda x: item_category_mapping(x)
    )

    session_category_events = session_events.merge(
        category_tree, how="left", left_on="category", right_on="categoryid"
    )[
        [
            "timestamp",
            "visitorid",
            "event",
            "itemid",
            "transactionid",
            "session",
            "category",
            "parentid",
        ]
    ]

    session_category_events["parentid"] = session_category_events["parentid"].fillna(-1)
    session_category_events = session_category_events.groupby(
        ["visitorid", "session"]
    ).filter(lambda x: "view" in x["event"].values)

    # 이력 많은 유저 필터링
    worthy_visitors = (
        session_category_events.groupby("visitorid")
        .size()
        .sort_values()[6000:]
        .index.tolist()
    )
    visitors_filtered = session_category_events[
        session_category_events["visitorid"].isin(worthy_visitors)
    ]

    final_df = visitors_filtered.groupby(["visitorid", "session"]).filter(
        lambda x: len(x) >= 5
    )

    item2idx = {
        item: idx for idx, item in enumerate(np.unique(final_df["itemid"].values))
    }

    category2idx = {
        category: idx
        for idx, category in enumerate(np.unique(final_df["category"].values))
    }

    pcategory2idx = {
        pcategory: idx
        for idx, pcategory in enumerate(np.unique(final_df["parentid"].values))
    }

    final_df["itemidx"] = final_df["itemid"].apply(lambda x: item2idx[x])
    final_df["categoryidx"] = final_df["category"].apply(lambda x: category2idx[x])
    final_df["pcategoryidx"] = final_df["parentid"].apply(lambda x: pcategory2idx[x])

    candidate_dict = (
        final_df[["itemidx", "pcategoryidx"]]
        .drop_duplicates()
        .groupby("pcategoryidx")["itemidx"]
        .apply(list)
        .to_dict()
    )

    return final_df, candidate_dict  # df


def data_augmentation(df, num_multiple):
    visitor_list = df["visitorid"].unique()
    max_session = df["session"].max()

    augmented_data = []
    timestamp_idx = df.columns.get_loc("timestamp")
    session_idx = df.columns.get_loc("session")

    for visitor in tqdm(visitor_list):
        visitor_df = df[df["visitorid"] == visitor]
        visitor_session = visitor_df["session"].unique()
        for visitor_ses in visitor_session:
            session_df = (
                visitor_df[visitor_df["session"] == visitor_ses].copy().to_numpy()
            )

            for times in range(num_multiple):
                max_session += 1
                session_df[:, timestamp_idx] = np.random.permutation(
                    session_df[:, timestamp_idx]
                )
                session_df[:, session_idx] = max_session
                augmented_data.extend(session_df)

    augmented_df = pd.DataFrame(augmented_data, columns=df.columns)
    return pd.concat([df, augmented_df], ignore_index=True)
