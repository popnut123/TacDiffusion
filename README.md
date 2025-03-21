# TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation
[[Paper]](https://arxiv.org/abs/2409.11047)
[[Video]](https://www.youtube.com/watch?v=dabpM4S9kbc&ab_channel=JeffWu)

![](readme/TacDiffusion_Overview.png)

## Overview

This repository is the official implementation of the paper [TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation](https://arxiv.org/abs/2409.11047) by Wu et al. (full citation below). 

In this work, we present a novel framework leveraging diffusion models to generate 6D wrench for tactile manipulation in high-precision robotic assembly tasks. Our approach, being the first force-domain diffusion policy, demonstrated excellent improved zero-shot transferability compared to prior work, by achieving an overall **95.7%** success rate in zero-shot transfer in experimental evaluations. Additionally, we investigate the trade-off between accuracy and inference speed and provide a practical guideline for optimal model selection. Further, we address the frequency misalignment between the diffusion policy and the real-time control loop with a dynamic system-based filter, significantly improving the task success rate by **9.15%**. Extensive experimental studies in our work underscore the effectiveness of our framework in real-world settings, showcasing a promising approach tackling high-precision tactile manipulation by learning diffusion-based transferable skills from expert policies containing primitive-switching logic. 

## Installation

The code was tested on Pop!_OS 22.04 LTS, which is equivalent to Ubuntu 22.04 LTS, with [Anaconda](https://www.anaconda.com/download) Python 3.9 and [PyTorch]((http://pytorch.org/)) 2.3.1. Higher versions should be possible with some accuracy difference. NVIDIA GPUs are needed for both training and testing.

---

1. Clone this repo:

    ```bash
    TacDiffusion_ROOT=/path/to/clone/TacDiffusion
    git clone https://github.com/popnut123/TacDiffusion.git $TacDiffusion_ROOT
    ```

2. Create an Anaconda environment or create your own virtual environment

    ```bash
    conda create -n TacDiffusion python=3.9
    conda activate TacDiffusion
    pip install -r requirements.txt
    conda install -c conda-forge eigenpy
    ```

3. Prepare training/testing data

    All training and testing data should be stored under `$TacDiffusion_ROOT/dataset/`.

    You can download the prepared datasets using the following link: [TacDiffusion Dataset](https://drive.google.com/drive/folders/10Ix8utcx51R8NejvGRF-ujWEGy5MK05R?usp=sharing).

## Training

1. To start a new training job with the default parameter settings, simply run the following:

    ```bash
    cd $TacDiffusion_ROOT
    python 1_model_train.py
    ```

    The result will be saved in `$TacDiffusion_ROOT/output/`, e.g., `TacDiffusion_model_512.pth`.

    You could then use tensorboard to visualize the training process via

    ```bash
    cd $TacDiffusion_ROOT
    tensorboard --logdir=logs --host=XX.XX.XX.XX
    ```

2. To optimize the inference speed, we recommend exporting models to the *ONNX* format. Simply modify the model name in the script `$TacDiffusion_ROOT/2_model_trans_pth_to_onnx.py` and run the following command:

    ```bash
    cd $TacDiffusion_ROOT
    python 2_model_trans_pth_to_onnx.py
    ```

    The converted model would be stored in `$TacDiffusion_ROOT/output/`, e.g., `TacDiffusion_model_512.onnx`.

---
***NOTE***
- The four trained models, each with different neuron configurations as discussed in our paper, are already provided in the folder `$TacDiffusion_ROOT/output/`.

- The implemented Diffusion Model (DDPM) in `$TacDiffusion_ROOT/helper_functions/models.py` is adapted from [Imitating-Human-Behaviour-w-Diffusion](https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion).

- The network architecture of the noise estimator is contructed as:

![](readme/TacDiffusion_noise_estimator.png)

## Testing

To test the model's theoretical performacne on recorded datasets, simply do the following: 

```bash
cd $TacDiffusion_ROOT
python 3_model_test.py
```

The testing results would be ploted in `$TacDiffusion_ROOT/figures/`.

## Deploying

Due to compatibility issues between the real-time kernel and the NVIDIA CUDA Toolkit, TacDiffusion should be implemented on a separate PC. The Franka Emika Panda robot manipulator can then be controlled via UDP communication by running the following command:

### CUDA PC side
```bash
cd $TacDiffusion_ROOT
python 4_model_remote_control.py
```
### real-time kernal PC side
**step 1**: install the real-timer kernal \
**step 2**: install the docker [https://docs.docker.com/engine/install/ubuntu/](tutorial)\
**step 3**: start the docker 
```bash
cd $TacDiffusion_ROOT/controller_in_docker
docker compose up
```
**step 4**: install the MongoDB service & connect to the local MongoDB database with the address "localhost:27017"\
**step 5**: using the query assistant of MongoDB with the command
```bash
db.getCollection("parameters").find({"name": "system"})
```
**step 6**: update the "robot_ip", "desk_name" and "desk_pwd" based on your own configuration\
**step 7**: close all the docker containers & boot your robot and lock the joint and release control in the robot desk\
**step 8**: restart the docker (detailed in step 3)\
**step 9**: homing the gripper, grasp the object and teach the peg-in-hole poses after pressing the user button (Note that: The `5_manipualtor_remote_control.py` and the dependent bbo folder and mios folder should be placed in the real-time PC.)
```bash
ipython3 -i 5_manipualtor_remote_control.py
call_method("localhost", 12000, "home_gripper") # code for homing the gripper
move_gripper(0.05) # code for releasing the gripper
grasp(0.01) # code for grasping the object

teach_location("localhost","peg_IL_test") # give the exp name 
teach_location("localhost","peg_IL_test_app") # record approach position
teach_location("localhost","peg_IL_test_hole") # record hole position
ctrl+z # finish the peg-in-hole poses teaching process
```
**step 10**: remote implement peg-in-hole experiments after changing the experiments name `peg_name_all` within the function `insert_auto_IL()` in `5_manipualtor_remote_control.py` with the releasing of the user button
```bash
ipython3 -i 5_manipualtor_remote_control.py
insert_auto_IL() # the manipulator would repeat insertion 3 times with the TacDiffusion_model_512. 
```

### Additional note about getting realtime kernel and cuda working on a single PC

It is possible to resolve the compatability issue of realtime kernel and nvidia driver. Simply following the guide [here](https://gist.github.com/FrankieWOO/8f5d0794512ae43a4efd7e068b19a4d8) if you have to deploy everything on a single GPU PC.

## Citation
Please cite the following if you use this repository in your publications:

```
@article{wu2024tacdiffusion,
  title={TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation},
  author={Wu, Yansong and Chen, Zongxie and Wu, Fan and Chen, Lingyun and Zhang, Liding and Bing, Zhenshan and Swikir, Abdalla and Knoll, Alois and Haddadin, Sami},
  journal={arXiv preprint arXiv:2409.11047},
  year={2024}
}
```

# License
No commercial use!
# Contact
For questions, please contact [Yansong Wu](mailto:yansong.wu@tum.de).
