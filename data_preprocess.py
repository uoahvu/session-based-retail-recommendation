import pandas as pd
import numpy as np
import datetime


def timestamp2datetime(x):
    if x > 10**10:
        x /= 1000  # Convert milliseconds to seconds
    return datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def session_flag(group):
    flag = 0
    flags = []
    for event in group["event"]:
        flags.append(flag)
        if event == "transaction":
            flag += 1
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

    return final_df  # df
