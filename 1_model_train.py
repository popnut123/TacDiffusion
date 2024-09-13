import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from helper_functions.data_split import RobotCustomDataset

# Set paths and hyperparameters
DATASET_PATH = "dataset"
SAVE_DATA_DIR = "output" 
os.makedirs(SAVE_DATA_DIR, exist_ok=True)

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)

n_epoch = 1500 
lrate = 1e-3 
device = "cuda" if torch.cuda.is_available() else "cpu"
n_hidden = 512 
batch_size = 4096 
n_T = 50
net_type = "fc"
drop_prob = 0.0
train_prop = 0.80
use_prev = True

Model_save_name = "model_512.pth"
state_dataset = 'robot_state_train.pkl'
action_dataset = 'robot_action_train.pkl'

# Load training and validation data
tf = transforms.Compose([])

torch_data_train = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="train", train_prop=train_prop,
    state_dataset=state_dataset, action_dataset=action_dataset
)
dataload_train = DataLoader(
    torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
)

torch_data_val = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="valid", train_prop=train_prop,
    state_dataset=state_dataset, action_dataset=action_dataset
)
dataload_val = DataLoader(
    torch_data_val, batch_size=batch_size, shuffle=False, num_workers=0
)

x_shape = torch_data_train.state_all.shape[1]
y_dim = torch_data_train.action_all.shape[1]

# Initialize the model and optimizer
nn_model = Model_mlp_diff_embed(
    x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type, use_prev=use_prev
).to(device)

model = Model_Cond_Diffusion(
    nn_model,
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    x_dim=x_shape,
    y_dim=y_dim,
    drop_prob=drop_prob,
    guide_w=0.0,
)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lrate)

# Set up TensorBoard logging
writer = SummaryWriter(log_dir=LOG_DIR)

# Main training loop
global_step = 1  
for ep in tqdm(range(n_epoch), desc="Epoch"):

    model.train()

    # Learning rate decay
    optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

    # Training loop
    pbar = tqdm(dataload_train)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.type(torch.FloatTensor).to(device)
        y_batch = y_batch.type(torch.FloatTensor).to(device)
        
        loss = model.loss_on_batch(x_batch, y_batch)
        optim.zero_grad()
        loss.backward()
        pbar.set_description(f"train loss: {loss.detach().item():.4f}")
        # Log training loss to TensorBoard
        writer.add_scalar('training_loss', loss.detach().item(), global_step)
        global_step += 1

        optim.step()

    if (ep + 1) % 5 == 0:  # Validate every 5 epochs
        model.eval()
        loss_val, n_batch_val = 0, 0
        with torch.no_grad():
            for x_batch_val, y_batch_val in tqdm(dataload_val, desc="Validation_Loss_MSE"):
                x_batch_val = x_batch_val.type(torch.FloatTensor).to(device)
                y_batch_val = y_batch_val.type(torch.FloatTensor).to(device)

                loss_val_inner = model.loss_on_batch(x_batch_val, y_batch_val)
                loss_val += loss_val_inner.detach().item()
                n_batch_val += 1

            avg_loss_val = loss_val / n_batch_val
            writer.add_scalar('validation_loss_mse', avg_loss_val, global_step)
            tqdm.write(f"Epoch {ep+1}, validation loss mse: {avg_loss_val:.4f}")

# Close TensorBoard writer
writer.close()

# Save the trained model
torch.save(model.state_dict(), os.path.join(SAVE_DATA_DIR, Model_save_name))
