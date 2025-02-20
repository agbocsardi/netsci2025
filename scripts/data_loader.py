import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def process_snowflake_exports(node_covar_list, y_label):
    """

    Function to process the data coming from snowflake. It produces a list of PyG Data objects, each corresponding to a snapshot (depends on what the level of analysis is in snowflake.

    """

    variable_node_attrs = pd.read_csv(
        "./data/netsci2025_variable_attributes.csv"
    ).fillna(0)

    unique_brands = variable_node_attrs["BRAND_NAT"].unique()
    brand_to_index = {brand: idx for idx, brand in enumerate(unique_brands)}

    # Load edge list and filter it to include only rows where both creator and publishing brands are in unique_brands
    edge_list = pd.read_csv("./data/netsci2025_content_share.csv")
    edge_list = edge_list[
        (edge_list["CREATOR_BRAND"].isin(unique_brands))
        & (edge_list["PUBLISHING_BRAND"].isin(unique_brands))
    ]

    if set(edge_list["PERIOD_OF_OBS"].unique()) == set(
        variable_node_attrs["DATE"].unique()
    ):
        snapshots = edge_list["PERIOD_OF_OBS"].unique()
    else:
        raise ("edge list and varcovars don't match on observations!")

    # Load constant node attributes and filter it to include only rows where brands are in unique_brands
    constant_node_attrs = pd.read_csv("./data/netsci2025_constant_node_attributes.csv")
    constant_node_attrs = constant_node_attrs[
        constant_node_attrs["BRAND_NAT"].isin(unique_brands)
    ].set_index("BRAND_NAT")

    # Create a pivot table to reshape the data
    edge_list_aggregated = edge_list.pivot_table(
        index=["PERIOD_OF_OBS", "CREATOR_BRAND", "PUBLISHING_BRAND"],
        columns="CONTENT_GROUP",
        values="NUM_ARTICLES_SHARED",
        fill_value=0,
    ).reset_index()

    data_list = []

    for snapshot in snapshots:
        edges = edge_list_aggregated[edge_list_aggregated["PERIOD_OF_OBS"] == snapshot]

        # edge_index creation
        creator_indices = [
            brand_to_index[creator] for creator in edges["CREATOR_BRAND"]
        ]

        receiver_indices = [
            brand_to_index[receiver] for receiver in edges["PUBLISHING_BRAND"]
        ]

        edge_index = torch.tensor([creator_indices, receiver_indices], dtype=torch.long)

        # edge attributes
        edge_attr_df = edges.drop(
            columns=["PERIOD_OF_OBS", "CREATOR_BRAND", "PUBLISHING_BRAND"], axis=1
        )
        edge_attrs = torch.from_numpy(edge_attr_df.to_numpy(np.float32))
        edge_attrs = (edge_attrs - edge_attrs.mean(axis=0)) / (edge_attrs.std(axis=0))

        edge_feature_name = {
            feature: idx for idx, feature in enumerate(edge_attr_df.columns)
        }

        # node table creation
        nodes = variable_node_attrs[variable_node_attrs["DATE"] == snapshot].set_index(
            "BRAND_NAT"
        )

        numeric_node_attrs = nodes.loc[:, node_covar_list]

        node_attrs = (
            numeric_node_attrs - numeric_node_attrs.mean()
        ) / numeric_node_attrs.std()
        node_attrs = node_attrs.join(constant_node_attrs["UIT_ANDERE"])

        one_hot_static_covars = pd.get_dummies(
            constant_node_attrs[["BRAND_PROFILE", "BUSINESS_MODEL"]], prefix="STATIC"
        )
        node_attrs = node_attrs.join(one_hot_static_covars)
        node_attrs["id"] = node_attrs.index.map(brand_to_index)
        node_attrs = node_attrs.sort_values("id").set_index("id")

        node_feature_name = {
            feature: idx for idx, feature in enumerate(node_attrs.columns)
        }
        x = torch.from_numpy(node_attrs.to_numpy(np.float32))

        # define target
        y_tensor = torch.tensor(nodes[y_label].values, dtype=torch.float).unsqueeze(1)
        eps = 1e-7
        y_tensor = torch.log(y_tensor + eps)

        data_list.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attrs, y=y_tensor)
        )

    return data_list, edge_feature_name, node_feature_name


def split_data(data_list, train_split, val_split=0.1, respect_temp_order=False):
    if not respect_temp_order:
        random.shuffle(data_list)

    num_snapshots = len(data_list)
    train_end_index = int(num_snapshots * train_split)
    val_end_index = train_end_index + int(num_snapshots * val_split)

    data_train = data_list[:train_end_index]
    data_val = data_list[train_end_index:val_end_index]
    data_test = data_list[val_end_index:]

    return data_train, data_val, data_test
