# TacDiffusion

## Overview

**TacDiffusion** is a project for training, validating, and deploying diffusion models. The repository includes various scripts and resources to facilitate the development and evaluation of these models. The `models.py` file located in the `helper_functions` directory is adapted from the GitHub repository maintained by Microsoft: [Imitating-Human-Behaviour-w-Diffusion](https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion).

### Directory Structure

- **dataset**: Contains the data used for training and testing the models.
- **figures**: Includes plots of testing results, generated after running `3_model_test.py`.
- **helper_functions**: Contains essential helper functions, including `models.py`, adapted from [Imitating-Human-Behaviour-w-Diffusion](https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion).
- **logs**: Stores training and validation loss records, which can be visualized using TensorBoard.
- **Output**: Directory for saving the trained diffusion models.

### Scripts

- **1_model_train.py**: Script for training diffusion models.
- **2_model_trans_pth_to_onnx.py**: Script for converting diffusion models from `.pth` format to `.onnx` format.
- **3_model_test.py**: Script for validating the performance of the trained diffusion models.
- **4_model_remote_control.py**: Script for remotely controlling the manipulator.

### Requirements

- **Python Version**: 3.9.19
- **requirements.txt**: Lists the required Python packages.

### Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Models**:
   ```bash
   python 1_model_train.py
   ```

3. **Convert Models**:
   ```bash
   python 2_model_trans_pth_to_onnx.py
   ```

4. **Validate Models**:
   ```bash
   python 3_model_test.py
   ```

5. **Remote Control**:
   ```bash
   python 4_model_remote_control.py
   ```
