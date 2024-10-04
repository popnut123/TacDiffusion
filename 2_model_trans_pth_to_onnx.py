import torch
import onnx
from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed

# Set device: use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model parameters
n_hidden = 512
x_shape = 36  # Assuming input tensor shape is [batch_size, 36]
y_dim = 6
net_type = "fc"
drop_prob = 0.0
use_prev = True

# Initialize the neural network model
nn_model = Model_mlp_diff_embed(
    x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type, use_prev=use_prev
).to(device)

# Initialize the conditional diffusion model
model = Model_Cond_Diffusion(
    nn_model,
    betas=(1e-4, 0.02),
    n_T=50,
    device=device,
    x_dim=x_shape,
    y_dim=y_dim,
    drop_prob=drop_prob,
    guide_w=0.0,
).to(device)

ModelName = "TacDiffusion_model_512"

# Load the trained model parameters
model.load_state_dict(torch.load(f"output/{ModelName}.pth"))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor for the model
dummy_input = torch.randn(1, x_shape).to(device)

# Export the model to ONNX format
torch.onnx.export(
    model, 
    dummy_input, 
    f"output/{ModelName}.onnx", 
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
