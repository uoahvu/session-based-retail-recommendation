import pandas as pd
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
    max_sessions = session_events.groupby("visitorid")["session"].max()

    # 각 `visitorid` 그룹에 대해, 최대 `session` 값을 가진 행을 삭제하기
    session_events = session_events[
        ~session_events.apply(
            lambda x: x["session"] == max_sessions[x["visitorid"]], axis=1
        )
    ]

    item_category = pd.concat(
        [
            item_properties_part1[item_properties_part1["property"] == "categoryid"],
            item_properties_part2[item_properties_part2["property"] == "categoryid"],
        ]
    )
    item_category_dict = item_category.set_index("itemid")["value"].to_dict()
    session_events["category"] = session_events["itemid"].apply(
        lambda x: item_category_dict.get(x, -1)
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

    return session_category_events  # df
