# Drone 6-DoF Pose Estimation

Deep learning-based visual estimation of 6-DoF drone pose from images, including dataset preparation, model training, evaluation, and comparison of neural network architectures.

## 1. Project Overview

This project focuses on estimating the 6-DoF pose of a drone from a single image using deep learning.

The pose consists of:

* **Position:** x, y, z
* **Orientation:** represented internally using a quaternion qx, qy, qz, qw

Although a 6-DoF pose has six degrees of freedom, quaternion orientation requires four numerical values because of the unit-norm constraint.

For interpretation and visualization, the predicted quaternion can be converted to Euler angles:

* Roll
* Pitch
* Yaw

The main objective is to develop and evaluate a neural-network-based visual pose estimation pipeline for indoor environments.

---

## 2. Project Objectives

The project aims to:

1. Develop a practical understanding of neural-network implementation and training using PyTorch.
2. Build an image-based regression pipeline for continuous pose estimation.
3. Implement a baseline 6-DoF pose estimator using a convolutional neural network.
4. Evaluate different neural-network architectures for the task.
5. Measure position and orientation estimation errors.
6. Analyze the limitations and performance of the proposed approach.

---

## 3. Project Timeline

**Project period:** September 1 – November 27, 2026

**Current date:** October 6, 2026

The remaining development time is limited, so the project prioritizes a functional end-to-end system over unnecessary additional technologies.

### Main milestones

| Target date | Milestone                                     |
| ----------- | --------------------------------------------- |
| Oct 9       | Practical PyTorch + MNIST                     |
| Oct 16      | CNN experiments + neural-network regression   |
| Oct 23      | ResNet-18 regression pipeline                 |
| Oct 30      | First real 6-DoF dataset/model                |
| Nov 6       | First quantitative pose-estimation results    |
| Nov 13      | Model comparison and improvements             |
| Nov 20      | Final experiments, metrics and visualizations |
| Nov 23–27   | Final documentation and repository cleanup    |

Weekends are not part of the planned schedule and are treated as additional buffer time.

---

## 4. Development Strategy

Because the theoretical foundations of machine learning are already familiar, the initial learning phase focuses primarily on **practical implementation**.

The strategy is:

```text
Brief concept review
        ↓
Implement in PyTorch
        ↓
Run experiment
        ↓
Modify parameters
        ↓
Deliberately break the model
        ↓
Analyze results
        ↓
Document what happened
        ↓
Apply the concept to the real project
```

The goal is not to reproduce tutorials without understanding them, but to develop the ability to independently build, train, debug and evaluate neural networks.

---

## 5. Learning Path

The practical learning path is:

```text
MNIST
  ↓
Basic neural network
  ↓
Training / validation / loss
  ↓
Controlled experiments
  ↓
CNN
  ↓
Neural-network regression
  ↓
ResNet-18
  ↓
6-DoF pose regression
```

Each exercise is intended to prepare a component that will later be used in the final system.

---

## 6. Target Model Architecture

The initial model will use a convolutional neural network as a visual feature extractor followed by a regression head.

The conceptual architecture is:

```text
                 Image
                   │
                   ▼
              ResNet-18
                   │
                   ▼
             Visual features
                   │
             Regression head
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Position         Orientation
       x, y, z          qx,qy,qz,qw
```

The intended final output is therefore:

```text
[x, y, z, qx, qy, qz, qw]
```

This contains seven numerical values representing six degrees of freedom.

The quaternion will be normalized and used internally for orientation. Euler angles will be obtained from the quaternion when needed for visualization, analysis, or interfaces that require roll, pitch and yaw.

### Candidate architectures

The project may compare:

* ResNet-18
* ResNet-34
* EfficientNet-B0

The exact comparison will depend on project progress, dataset size and available computational resources.

---

## 7. Development Environment

### Local development machine

Current development laptop:

* HP Pavilion 15-cw1xxx
* AMD Ryzen 5 3500U
* 24 GB RAM
* AMD Radeon Vega 8
* No NVIDIA GPU

The laptop will primarily be used for:

* Development
* Dataset preparation
* Small experiments
* Debugging
* Initial training
* Evaluation
* Visualization

Heavier training may be performed using Google Colab or an available NVIDIA GPU at the university/INAOE.

### Operating system

* Ubuntu 24.04 LTS
* Python 3.12.3

### Main Python libraries

* PyTorch
* torchvision
* NumPy
* OpenCV
* pandas
* matplotlib
* scikit-learn
* tqdm
* Pillow

Dependencies are listed in `requirements.txt`.

---

## 8. Python Virtual Environment

The project uses a Python virtual environment named `.venv`.

The `.venv` directory is intentionally not tracked by Git.

### Activate the environment

After opening a new terminal or restarting the computer:

```bash
cd ~/drone-6dof-pose-estimation
source .venv/bin/activate
```

After activation, the terminal should show something similar to:

```text
(.venv) ricardo@ricardo-HP-Pavilion-Laptop-15-cw1xxx:~/drone-6dof-pose-estimation$
```

### Verify the environment

```bash
which python
python --version
```

Expected result:

```text
/home/ricardo/drone-6dof-pose-estimation/.venv/bin/python
Python 3.12.3
```

It is also useful to verify PyTorch:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Deactivate the environment

When finished working:

```bash
deactivate
```

Deactivating the environment does **not** delete it.

The environment can be activated again the next time the project is opened.

---

## 9. Initial Environment Verification

The local Python environment has been successfully configured.

The following libraries have been installed and tested:

* PyTorch
* torchvision
* NumPy
* OpenCV
* pandas
* matplotlib
* scikit-learn
* tqdm
* Pillow

A basic ResNet-18 forward pass has been successfully executed locally.

### Local environment

```text
PyTorch: 2.14.1+cu130
Device: CPU
CUDA available: False
```

The local laptop does not have an NVIDIA CUDA-compatible GPU. The AMD Vega 8 graphics processor is therefore not being used for CUDA acceleration.

This is not a problem for development and small experiments.

---

## 10. Local ResNet-18 Verification

A basic ResNet-18 model was successfully created and executed locally.

Test input:

```text
torch.Size([1, 3, 224, 224])
```

Standard ResNet-18 output:

```text
torch.Size([1, 1000])
```

This confirmed that PyTorch and torchvision are correctly installed and that the model can execute locally.

---

## 11. Neural Network Training Verification

The next verification step went beyond simply executing ResNet-18.

A modified ResNet-18 regression model was tested with:

```text
Input:
[1, 3, 224, 224]

Output:
[1, 6]
```

The model was successfully tested through the complete basic training cycle:

```text
Input image
     ↓
Forward pass
     ↓
Prediction
     ↓
Loss calculation
     ↓
Backpropagation
     ↓
Gradient calculation
     ↓
Weight update
```

Observed results included:

```text
Loss: 2.1600475311279297
Gradiente calculado: True
Pesos actualizados: True
```

This confirms that the local environment is capable of performing not only inference but also the fundamental training operations required for the project.

### Important clarification

The six-output experiment was a **technical verification of the regression/training pipeline**.

It does **not** yet represent the final 6-DoF pose representation.

The final pose representation currently planned is:

```text
Position:
x, y, z

Orientation:
qx, qy, qz, qw
```

Therefore, the intended final regression output is:

```text
[x, y, z, qx, qy, qz, qw]
```

The exact loss function and normalization strategy for this representation will be defined during the 6-DoF regression stage.

---

## 12. Local Laptop vs Google Colab

The basic ResNet-18 regression/training experiment has been successfully tested in both environments.

| Test                 | Laptop | Google Colab |
| -------------------- | -----: | -----------: |
| PyTorch              |      ✓ |            ✓ |
| torchvision          |      ✓ |            ✓ |
| ResNet-18            |      ✓ |            ✓ |
| Input `1×3×224×224`  |      ✓ |            ✓ |
| Regression output    |      ✓ |            ✓ |
| Loss calculation     |      ✓ |            ✓ |
| Gradient calculation |      ✓ |            ✓ |
| Weight update        |      ✓ |            ✓ |
| GPU acceleration     |     No |  ✓ NVIDIA T4 |
| Main device          |    CPU |         CUDA |

The environments currently use different PyTorch versions:

```text
Laptop:
PyTorch 2.14.1+cu130

Google Colab:
PyTorch 2.11.0+cu130
```

This difference has not prevented the experiments from running successfully.

Before final large-scale training, the project environment and relevant package versions should be documented or pinned sufficiently to improve reproducibility.

---

## 13. Repository Structure

The project structure will grow as the methodology becomes defined.

Planned structure:

```text
drone-6dof-pose-estimation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── scripts/
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Directories and files will be created only when they are needed by the project workflow.

---

## 14. Dataset and Ground Truth

The final dataset will contain images associated with known drone/camera poses.

Each pose is expected to contain:

```text
Position:
x, y, z

Orientation:
qx, qy, qz, qw
```

Before constructing the final dataset, the following must be defined:

* Camera used for image acquisition
* Coordinate system
* Position reference frame
* Orientation convention
* Camera intrinsics
* Camera/drone relationship
* Ground-truth pose acquisition method
* Image/pose synchronization
* Training/validation/test split

Possible tools and methods will be evaluated rather than added automatically.

In particular, COLMAP will only be incorporated if it is appropriate for the chosen ground-truth generation methodology.

---

## 15. Pose Representation

A 6-DoF pose contains:

```text
3 DoF position
+
3 DoF orientation
```

Position:

```text
x, y, z
```

Orientation may be expressed using Euler angles:

```text
roll, pitch, yaw
```

However, the model is currently planned to use a quaternion:

```text
qx, qy, qz, qw
```

because quaternions avoid several problems associated with directly regressing Euler angles, including angular discontinuities and gimbal-lock-related issues.

For interpretation:

```text
Quaternion
    ↓
Euler conversion
    ↓
Roll / Pitch / Yaw
```

Euler angles will therefore be useful for visualization and human interpretation, but they will not necessarily be the primary representation used during training.

---

## 16. Planned Training Pipeline

The final training pipeline is expected to follow:

```text
Images
   ↓
Dataset
   ↓
DataLoader
   ↓
Image preprocessing / augmentation
   ↓
CNN backbone
   ↓
Feature representation
   ↓
Regression head
   ↓
Position + orientation
   ↓
Pose loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Updated model
```

The training objective will combine position and orientation errors.

Conceptually:

```text
L = L_position + λ L_rotation
```

The exact formulation of the rotation loss and the value of λ will be investigated experimentally.

---

## 17. Evaluation

The final system will be evaluated using quantitative metrics.

Planned measurements include:

* Position error in meters
* Orientation error in degrees
* Training loss
* Validation loss
* Test-set performance
* Inference time
* Model size / computational cost when relevant

For orientation, the primary evaluation should use a rotation-aware angular error rather than simply subtracting Euler angles.

Results should be reported consistently across different architectures.

---

## 18. Model Comparison

If project progress allows, multiple architectures will be evaluated using the same dataset, training protocol and evaluation metrics.

Potential comparison:

| Model           | Position Error | Orientation Error | Inference Time | Computational Cost |
| --------------- | -------------: | ----------------: | -------------: | -----------------: |
| ResNet-18       |              — |                 — |              — |                  — |
| ResNet-34       |              — |                 — |              — |                  — |
| EfficientNet-B0 |              — |                 — |              — |                  — |

The first priority is obtaining a reliable baseline. Additional architectures will only be evaluated if they do not compromise the completion of the end-to-end system.

---

## 19. Experiment Log

Important experiments will be documented throughout the project.

For each experiment, record:

* Date
* Objective
* Model
* Dataset
* Parameters
* Training configuration
* Loss
* Metrics
* Observations
* Conclusions

This log will also serve as a basis for the final project report.

---

## 20. Current Status — October 6, 2026

### Environment

* [x] Git repository created
* [x] GitHub repository configured
* [x] SSH authentication configured
* [x] Python virtual environment created
* [x] PyTorch installed
* [x] torchvision installed
* [x] Required Python libraries installed
* [x] Local environment verified
* [x] ResNet-18 tested locally
* [x] `requirements.txt` created

### Practical PyTorch

* [x] Basic forward pass tested
* [x] Regression output tested
* [x] Loss calculation tested
* [x] Backpropagation tested
* [x] Gradients verified
* [x] Weight updates verified
* [x] Same basic training pipeline tested in Google Colab
* [x] Local CPU execution verified
* [x] Google Colab GPU execution verified

### Learning phase

* [ ] MNIST practical experiment
* [ ] Controlled parameter experiments
* [ ] Loss/accuracy visualization
* [ ] CNN experiment
* [ ] Neural-network regression experiment
* [ ] ResNet-18 regression experiment

### Real project

* [ ] Define ground-truth acquisition method
* [ ] Define coordinate systems
* [ ] Define camera configuration
* [ ] Build real dataset
* [ ] Train first 6-DoF model
* [ ] Evaluate pose errors
* [ ] Compare architectures
* [ ] Generate final results
* [ ] Final documentation

---

## 21. Current Next Step

The environment and basic PyTorch training pipeline have already been verified.

The next stage is **practical neural-network training using MNIST**.

The MNIST experiment will be used to practice:

* Dataset loading
* DataLoader
* Model definition
* Forward pass
* Loss
* Backpropagation
* Optimizer
* Training loop
* Validation
* Metrics
* Experimentation
* Visualization

The experiment will intentionally include parameter modifications and controlled failures rather than simply reproducing a tutorial.

After this stage, the project will progress toward CNNs, regression, ResNet-18 and finally the real 6-DoF pose-estimation problem.
