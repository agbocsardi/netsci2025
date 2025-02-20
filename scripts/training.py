import torch
import torch.nn.functional as F
from data_loader import process_snowflake_exports, split_data
from model_definition import GATModel
from torch_geometric.loader import DataLoader

learning_rate = 0.01
epochs = 1000
batch_size = 2

node_attrs_used = [
    "N_ARTICLES_PUBLISHED",
    "SUM_TOTAL_TIME_SECONDS",
    "TOTAL_SUBS",
    "NEW_SUBS",
    "LOST_SUBS",
    # "PAGEVIEWS",
]
target_var = "PAGEVIEWS"

graph_list, edge_feature_dict, node_feature_dict = process_snowflake_exports(
    node_attrs_used, target_var
)

train_data, val_data, test_data = split_data(
    data_list=graph_list, train_split=0.6, val_split=0.2
)

train_loader = DataLoader(train_data, batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size, shuffle=False)

model = GATModel(
    num_node_features=len(node_feature_dict), num_edge_features=len(edge_feature_dict)
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    model.train()
    train_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.num_graphs

    avg_train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data in val_loader:
            output = model(data)
            loss = F.mse_loss(output, data.y)
            val_loss += loss.item() * data.num_graphs

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(
        f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

# Evaluate on test set
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
model.eval()
# test_loss = 0
#
# with torch.no_grad():
#     for data in test_loader:
#         output = model(data)
#         loss = F.mse_loss(output, data.y)
#         test_loss += loss.item() * data.num_graphs
#
# avg_test_loss = test_loss / len(test_loader.dataset)
# print(f"Test Loss: {avg_test_loss:.4f}")
# Initialize accumulators
total_mae = 0.0
total_mse = 0.0
residual_sum_of_squares = 0.0
total_variance = 0.0

y_values = []

with torch.no_grad():
    for data in test_loader:
        output = model(data)

        # Collect true y values for descriptives
        y_values.extend(data.y.view(-1).cpu().numpy())

        # Calculate errors and losses
        error = output - data.y
        mae_batch = torch.abs(error).mean().item()
        mse_batch = F.mse_loss(output, data.y).item()

        total_mae += mae_batch * data.num_graphs
        total_mse += mse_batch

        # Calculate R-squared components
        y_mean = data.y.mean().item()
        residual_sum_of_squares += (error**2).sum().item()
        total_variance += ((data.y - y_mean) ** 2).sum().item()

# Calculate averages and other metrics
num_samples = len(test_loader.dataset)
avg_mae = total_mae / num_samples
avg_mse = total_mse / num_samples
r_squared = 1 - (residual_sum_of_squares / total_variance)

# Convert y_values to a tensor for easy statistics calculation
y_tensor = torch.tensor(y_values, dtype=torch.float32)

# Descriptive statistics
y_mean = y_tensor.mean().item()
y_std = y_tensor.std().item()
y_min = y_tensor.min().item()
y_max = y_tensor.max().item()

# Print results
print(f"Mean of true values: {y_mean:.4f}")
print(f"Standard Deviation of true values: {y_std:.4f}")
print(f"Min of true values: {y_min:.4f}")
print(f"Max of true values: {y_max:.4f}")

print(f"Average MAE: {avg_mae:.4f}")
print(f"Average MSE (Test Loss): {avg_mse:.4f}")
print(f"R-squared: {r_squared:.4f}")
