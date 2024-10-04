import os
import torch
import time
import xlsxwriter
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
from torchvision import transforms
from helper_functions.data_split import RobotCustomDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set paths and directories
DATASET_PATH = "dataset"
SAVE_FIGURE_DIR = "figures/TacDiffusion_model_512"
FIGURE_ACTION_DIR = os.path.join(SAVE_FIGURE_DIR, "figures_action")
FIGURE_state_DIR = os.path.join(SAVE_FIGURE_DIR, "figures_state")
FIGURE_ERROR_DIR = os.path.join(SAVE_FIGURE_DIR, "figures_error")
interval_length = 5000  # Length of valid data

# Set device: use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')

# Dataset files
state_dataset = 'robot_state_test.pkl'
action_dataset = 'robot_action_test.pkl'
print(f'state_dataset: {state_dataset}')
print(f'action_dataset: {action_dataset}')

# Model file
model_name = 'output/TacDiffusion_model_512.onnx'
print(f'model_name: {model_name}')

# Load datasets
tf = transforms.Compose([])
torch_data_test = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="test", train_prop=0.01,
    state_dataset=state_dataset, action_dataset=action_dataset
)

# Extract input and output dimensions
x_shape = torch_data_test.state_all.shape[1]
y_dim = torch_data_test.action_all.shape[1]
y_true = torch_data_test.action_all
print(f'y_dim: {y_dim}')
print('data import success!')

# Enable graph optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Create ONNX inference session
ort_session = ort.InferenceSession(model_name, sess_options)

# Determine number of intervals in the dataset
dim_validation = torch_data_test.state_all.shape[0]
num_intervals = dim_validation // interval_length

# Labels for predictions and ground truth
label_pred = ['f_x_pred', 'f_y_pred', 'f_z_pred', 'tau_x_pred', 'tau_y_pred', 'tau_z_pred']
label_true = ['f_x_true', 'f_y_true', 'f_z_true', 'tau_x_true', 'tau_y_true', 'tau_z_true']
data_label = ['f_x_ext', 'f_y_ext', 'f_z_ext', 
              'tau_x_ext', 'tau_y_ext', 'tau_z_ext', 
              'f_x_in', 'f_y_in', 'f_z_in', 
              'tau_x_in', 'tau_y_in', 'tau_z_in', 
              'v_x', 'v_y', 'v_z',
              'w_x', 'w_y', 'w_z']

subplot_label = ['wrench_ext_f',
                 'wrench_ext_tau',
                 'wrench_inner_f',
                 'wrench_inner_tau',
                 'linear velocity',
                 'angular velocity']

# Initialize DataFrame to store error metrics and inference speed
error_df = pd.DataFrame(columns=["Figure Name", "MAE", "RMSE", "Inference Speed (samples/second)"])

# Lists to store all ground truth, predictions, and inference speeds for overall error calculation
all_y_true = []
all_y_pred = []
all_inference_speeds = []

# Create directories to save figures if they do not exist
if not os.path.exists(SAVE_FIGURE_DIR):
    os.makedirs(SAVE_FIGURE_DIR)
if not os.path.exists(FIGURE_ACTION_DIR):
    os.makedirs(FIGURE_ACTION_DIR)
if not os.path.exists(FIGURE_state_DIR):
    os.makedirs(FIGURE_state_DIR)
if not os.path.exists(FIGURE_ERROR_DIR):
    os.makedirs(FIGURE_ERROR_DIR)

# Iterate through intervals and perform inference
for interval_idx in range(num_intervals):
    start_idx = interval_idx * interval_length
    end_idx = start_idx + interval_length
    idxs = range(start_idx, end_idx)
    y_pred = np.zeros((interval_length, y_dim))  # Array to store prediction results

    start_time = time.time()

    # Perform inference on each sample in the interval
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            x_eval = torch.Tensor(torch_data_test.state_all[idx]).type(torch.FloatTensor).to(device)
            x_eval_ = x_eval.repeat(1, 1).cpu().numpy()
            y_pred_ = ort_session.run(['output'], {'input': x_eval_})[0]
            y_pred[i] = y_pred_

    end_time = time.time()
    inference_time = end_time - start_time
    inference_speeds = interval_length / inference_time
    print(f"Interval {interval_idx}: inference speeds: {inference_speeds} sample/second")

    # Calculate errors for the current interval
    y_true_interval = y_true[start_idx:end_idx]
    mae = mean_absolute_error(y_true_interval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_interval, y_pred))

    # Add the current interval's error and inference speed to the DataFrame
    figure_saved_name = f"action_data_{start_idx}_{end_idx}.png"
    new_row = pd.DataFrame({
        "Figure Name": [figure_saved_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "Inference Speed (samples/second)": [inference_speeds]
    })
    error_df = pd.concat([error_df, new_row], ignore_index=True)

    # Add to the list for overall error calculation
    all_y_true.append(y_true_interval)
    all_y_pred.append(y_pred)
    all_inference_speeds.append(inference_speeds)

    # Plot and save the action data figures
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    for i in range(6):
        row, col = divmod(i, 2)
        axs[row, col].plot(y_pred[:, i], label=label_pred[i], linestyle='-')
        axs[row, col].plot(y_true_interval[:, i], label=label_true[i], linestyle='--')

        axs[row, col].set_xlabel('Index')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(f'Line Plot of {label_pred[i]} and {label_true[i]}')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_ACTION_DIR, figure_saved_name))
    plt.close(fig)
    print(f"Saved figure {figure_saved_name} in {FIGURE_ACTION_DIR}")

    # Plot and save the state data figures
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    for k in range(6):
        row, col = divmod(k, 2)  # Calculate row and column index
        start_col = k * 3

        # Plot up to three columns of state data in the current subplot
        for l in range(3):
            col_index = start_col + l
            if col_index < 18:  # Ensure col_index is within bounds
                axs[row, col].plot(torch_data_test.state_all[start_idx:end_idx, col_index], label=f'{data_label[col_index]}', linestyle='-')

        axs[row, col].set_xlabel('Index')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(f'{subplot_label[k]}')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()  # Adjust subplots to fit into figure area

    state_figure_saved_name = f"state_data_{start_idx}_{end_idx}.png"
    plt.savefig(os.path.join(FIGURE_state_DIR, state_figure_saved_name))
    plt.close(fig)
    print(f"Saved state figure {state_figure_saved_name} in {FIGURE_state_DIR}")

# Calculate overall error metrics
all_y_true = np.vstack(all_y_true)
all_y_pred = np.vstack(all_y_pred)
overall_mae = mean_absolute_error(all_y_true, all_y_pred)
overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_avg_speed = np.mean(all_inference_speeds)

# Add overall error metrics and average inference speed to the DataFrame
overall_row = pd.DataFrame({
    "Figure Name": ["Overall"],
    "MAE": [overall_mae],
    "RMSE": [overall_rmse],
    "Inference Speed (samples/second)": [overall_avg_speed]
})
error_df = pd.concat([error_df, overall_row], ignore_index=True)

# Save the error metrics to an Excel file
error_save_path = os.path.join(SAVE_FIGURE_DIR, "Error_Saved.xlsx")
with pd.ExcelWriter(error_save_path, engine='xlsxwriter') as writer:
    error_df.to_excel(writer, index=False)
print("All figures and error metrics saved successfully.")

# Plot and save bar charts for the errors
num_plots = (len(error_df) - 1) // 10 + 1
for i in range(num_plots):
    start_idx = i * 10
    end_idx = min((i + 1) * 10, len(error_df) - 1)
    fig, ax = plt.subplots(figsize=(12, 8))
    error_df[start_idx:end_idx].plot(kind='bar', x='Figure Name', y=['MAE', 'RMSE'], ax=ax)
    plt.xticks(rotation=90)
    plt.title(f'MAE and RMSE for each interval (Part {i + 1})')
    plt.xlabel('Figure Name')
    plt.ylabel('Error')
    plt.tight_layout()
    error_bar_plot_name = f"Error_Bar_Plot_Part_{i + 1}.png"
    plt.savefig(os.path.join(FIGURE_ERROR_DIR, error_bar_plot_name))
    plt.close(fig)
    print(f"Saved error bar plot {error_bar_plot_name} in {FIGURE_ERROR_DIR}")
