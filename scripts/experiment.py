import optuna
import torch
import torch.nn.functional as F
from aim.optuna import AimCallback
from data_loader import process_snowflake_exports, split_data
from model_definition import GATModel
from torch_geometric.loader import DataLoader

EXPERIMENT_NAME = "GAT on Pageviews"

aim_callback = AimCallback(
    as_multirun=True, metric_name="avg_val_loss", experiment_name=EXPERIMENT_NAME
)


def initialize_model(num_node_features, num_edge_features, trial):
    return GATModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_size=trial.suggest_int("hidden_size", 4, 32, step=4),
        num_attention_heads=trial.suggest_int("num_heads", 4, 32, step=4),
        dropout=trial.suggest_float("dropout", 0, 1),
    )


@aim_callback.track_in_aim()
def objective(trial):
    # defining the model setup used in the experiment
    node_attrs_used = [
        "N_ARTICLES_PUBLISHED",
        "SUM_TOTAL_TIME_SECONDS",
        "TOTAL_SUBS",
        "NEW_SUBS",
        "LOST_SUBS",
    ]
    target_var = "PAGEVIEWS"

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 1, 3)
    epochs = trial.suggest_int("epochs", 100, 1000)

    graph_list, edge_feature_dict, node_feature_dict = process_snowflake_exports(
        node_attrs_used, target_var
    )

    # aim_callback.experiment.track(
    #     Text(str(node_feature_dict)), name="node attributes used"
    # )
    aim_callback.experiment["node_attrs_used"] = list(node_feature_dict.keys())
    aim_callback.experiment["node_attr_mapping"] = str(node_feature_dict)

    aim_callback.experiment["edge_attrs_used"] = list(edge_feature_dict.keys())
    aim_callback.experiment["edge_attr_mapping"] = str(edge_feature_dict)

    aim_callback.experiment["target_variable"] = target_var

    # aim_callback.experiment.track(
    #     Text(str(edge_feature_dict)), name="edge attributes used"
    # )
    # aim_callback.experiment.track(Text(target_var), name="target variable")
    # aim_callback.experiment.track(learning_rate, name="learning_rate")
    # aim_callback.experiment.track(epochs, name="epochs")

    train_data, val_data, test_data = split_data(
        data_list=graph_list, train_split=0.6, val_split=0.2
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = initialize_model(
        num_node_features=len(node_feature_dict),
        num_edge_features=len(edge_feature_dict),
        trial=trial,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data in val_loader:
            output = model(data)
            loss = F.mse_loss(output, data.y)
            val_loss += loss.item() * data.num_graphs

    avg_val_loss = val_loss / len(val_loader.dataset)

    aim_callback.experiment.track(avg_val_loss, name="avg_val_loss")

    return avg_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500, callbacks=[aim_callback])

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
