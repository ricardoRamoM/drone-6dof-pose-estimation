# Drone 6-DoF Pose Estimation

Deep-learning-based visual estimation of camera pose from indoor images, using COLMAP-generated pose estimates as initial labels and ResNet-18 as the baseline regression model.

**Project period:** September 1 – November 27, 2026  
**Status updated:** October 14, 2026

---

## 1. Project Overview

This project investigates whether a convolutional neural network can estimate the 6-DoF pose associated with an indoor image.

A 6-DoF pose describes:

- **Position:** \(x, y, z\)
- **Orientation:** three rotational degrees of freedom, represented by a unit quaternion \(q_x, q_y, q_z, q_w\)

The planned regression output contains seven numerical values:

```text
[x, y, z, qx, qy, qz, qw]
```

A quaternion uses four values to represent a three-degree-of-freedom rotation. Predicted quaternions must be normalized, and orientation evaluation must account for the fact that \(q\) and \(-q\) represent the same rotation.

The current dataset comes from twelve indoor videos recorded with a Parrot Bebop 2. Frames were extracted and processed together with COLMAP, and the exported camera poses are being used as **estimated pose labels**.

The first milestone is a reproducible image-to-pose pipeline. Metric alignment, independent validation, and the transformation from camera pose to drone-body pose are separate tasks that remain to be addressed.

---

## 2. Project Objectives

The project aims to:

1. Review and apply visual-pose-estimation methods and the tools required to implement them.
2. Build an image dataset from Bebop 2 video sequences and associate registered images with COLMAP-estimated camera poses.
3. Inspect reconstruction quality and document the limitations of using COLMAP poses as training labels.
4. Implement a ResNet-18 baseline that regresses position and orientation from an image.
5. Evaluate position and orientation errors on held-out video sequences.
6. Investigate how to align the reconstruction with a physical coordinate frame and metric scale.
7. Explore inference or integration with ROS and the Bebop, if time and model quality allow.
8. Compare additional neural-network architectures only if the baseline is complete and the schedule permits.

The core deliverable is a working, evaluated image-to-pose pipeline and a clear account of its limitations. Additional architectures, Gaussian Splatting, and a second-camera dataset are optional extensions.

---

## 3. Project Timeline

The project runs from September 1 to November 27, 2026. This schedule reflects progress as of October 14 and prioritizes completing an end-to-end baseline.

| Target dates | Milestone | Status |
|---|---|---|
| Sep 1–25 | State-of-the-art review and tool familiarization | Completed / continued as needed |
| Sep 21–Oct 13 | Capture indoor sequences, extract frames, run COLMAP | Twelve videos captured; frames extracted; feature extraction and exhaustive matching completed; joint reconstruction and pose export completed |
| Oct 13–16 | Inspect labels, document conventions, create a reproducible split | In progress |
| Oct 16–20 | Implement image/pose `Dataset`, `DataLoader`, and preprocessing | Planned |
| Oct 16–23 | Implement the seven-output ResNet-18 training/evaluation pipeline | Planned |
| Oct 23–30 | Run the first real-data training experiment | Planned |
| Oct 30–Nov 6 | Debug, tune, and establish quantitative baseline results | Planned |
| Nov 6–13 | Optional architecture comparison or additional data experiment | Optional |
| Nov 9–20 | Test image-to-pose inference and explore ROS/Bebop integration | Planned; scope depends on baseline progress |
| Nov 16–23 | Final validation, plots, metrics, and limitations | Planned |
| Nov 23–27 | Final report, README, repository cleanup, and delivery | Planned |

Dates may overlap where tasks can proceed independently. The first priority is to complete the COLMAP-label dataset and establish a reliable ResNet-18 baseline. Weekends are not included in the main work estimate and provide schedule buffer.

---

## 4. Development Strategy

The project follows an implementation-first strategy:

1. Review a concept when needed.
2. Implement it in PyTorch.
3. Run a controlled experiment.
4. Inspect predictions, losses, gradients, and metrics.
5. Modify parameters or deliberately test failure cases.
6. Document the outcome.
7. Apply the lesson to the real pose-estimation task.

A separate MNIST exercise is **not a mandatory milestone**. The local and Colab checks already verified the basic forward pass, regression output, loss computation, gradient calculation, and optimizer update. The next learning steps should use the actual image-pose dataset wherever practical.

---

## 5. Current Methodology

The current planned pipeline is:

```text
12 Parrot Bebop 2 video sequences
              |
              v
        Extract frames
              |
              v
     COLMAP feature extraction
              |
              v
       Exhaustive matching
              |
              v
     Joint structure-from-motion
              |
              v
   Inspect reconstruction and registered images
              |
              v
   Export camera poses to a CSV file
              |
              v
   Validate image/pose associations and conventions
              |
              v
  Split data by complete video sequence
              |
              v
   PyTorch Dataset and DataLoader
              |
              v
   ResNet-18 image-to-pose regression
              |
              v
  Evaluate on held-out video sequences
              |
              v
  Optional coordinate alignment / ROS inference
```

The current pose labels come from COLMAP processing the same Bebop images used as model inputs. They are therefore not independent ground-truth measurements.

The current project does **not** use Bebop odometry as the primary label source and does not yet use Vicon measurements. Independent validation remains desirable and must be reported separately from agreement with COLMAP.

---

## 6. Pose Labels and Their Limitations

### 6.1 Current label file

The exported pose table is:

```text
data/poses_colmap.csv
```

It contains one row per registered image and is expected to include these fields:

```text
image_name, x, y, z, qx, qy, qz, qw
```

Check the actual CSV header before running scripts that depend on exact column names.

### 6.2 Current validation results

The pose CSV passed the following basic checks:

| Check | Result |
|---|---:|
| Images with exported poses | 609 |
| Unique image names | 609 |
| CSV images missing from the image folder | 0 |
| Rows with invalid numeric values | 0 |
| Non-normalized quaternions | 0 |

The exported coordinate ranges are currently:

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| \(x\) | -2.53 | 4.91 |
| \(y\) | -2.50 | 1.96 |
| \(z\) | -3.49 | 5.23 |

These values are expressed in the reconstructed coordinate system as exported by the current pose-conversion script. They must **not** be assumed to be meters.

The number of registered images varies by sequence. For example, the complete-route sequence has 140 registered images and the circular-route sequence has 94. A sequence with fewer registered images is not automatically invalid, but reconstruction coverage and quality should be inspected.

### 6.3 Important interpretation caveats

- **Estimated labels:** COLMAP poses are reconstruction estimates, not independently measured ground truth.
- **Monocular scale:** a monocular reconstruction generally has an unknown global scale unless metric information is introduced.
- **Coordinate frame:** COLMAP's origin and axes are not automatically aligned with the room or ROS.
- **Pose convention:** COLMAP's stored image extrinsics normally represent a world-to-camera transformation. The camera center in the reconstruction frame is computed from the rotation and translation. Verify that the export code makes `x, y, z` and the quaternion consistently describe the intended camera pose.
- **Camera versus drone:** the image pose describes the camera. It is not automatically the pose of the Bebop body.
- **Training target:** a model trained on these labels learns to reproduce COLMAP's pose estimates and their errors. Good test performance against these labels does not by itself prove accurate real-world localization.

Before final training, document the exact meaning of the exported position and quaternion, quaternion order, rotation direction, and coordinate frame.

---

## 7. Coordinate-Frame Plan

The intended physical frame for future alignment is provisionally defined as follows:

- **Origin:** the floor-level corner where the nets meet.
- **X axis:** horizontal, pointing into the space in the initial viewing direction of the Bebop.
- **Y axis:** horizontal and perpendicular to X, following the other direction of the space.
- **Z axis:** vertical and positive upward.
- **Units:** meters, after a valid metric-scale alignment.

This is the target convention, not the current COLMAP frame.

The exact physical location of the camera in the first frame has not yet been established. The first image must not automatically be assigned position \((0,0,0)\) merely because recording began near the reference corner.

For positions, the intended similarity transform is:

\[
\mathbf{p}_{target} = s R_a \mathbf{p}_{COLMAP} + \mathbf{t}_a
\]

where \(s\) is the scale factor, \(R_a\) aligns the axes, and \(\mathbf{t}_a\) establishes the target origin. A consistent transformation must also be applied to orientations.

Potential sources for alignment include measured room dimensions and known camera positions, or independent position references obtained using the INAOE Vicon system. A single known distance can constrain scale but does not determine the complete scale/rotation/translation alignment on its own.

**Status:** physical alignment, metric scale, and independent reference measurements are pending. Do not describe the current CSV as metric ground truth.

---

## 8. Dataset and Capture Status

### 8.1 Source videos

Twelve indoor videos were recorded with the Parrot Bebop 2. Frames were extracted from all twelve videos and used for the joint COLMAP workflow.

The source sequence names are:

```text
bebop_001_desde_piso
bebop_002_avance_bajo
bebop_003_elevacion
bebop_004_avance_elevacion
bebop_005_pared_1
bebop_006_pared_2
bebop_007_avance_giro
bebop_008_diagonal
bebop_009_esquina_opuesta
bebop_010_recorrido_lateral
bebop_011_recorrido_circular
bebop_012_recorrido_completo
```

The extracted frame folders are under:

```text
data/frames/
```

Video and frame naming should remain consistent because the video identifier is used to assign all frames from a sequence to the same split.

### 8.2 Current data artifacts

| Artifact / stage | Status |
|---|---|
| Twelve Bebop videos captured | Complete |
| Frames extracted from all twelve videos | Complete |
| COLMAP feature extraction | Complete |
| COLMAP exhaustive matching | Complete |
| Joint reconstruction | Complete |
| Camera poses exported to `data/poses_colmap.csv` | Complete |
| Basic CSV/image/quaternion validation | Complete |
| Sequence-level train/validation/test CSV | Create or verify next |
| PyTorch dataset and loader | Pending |
| Model training and evaluation | Pending |

### 8.3 Inclusion rule

Only images with a valid exported pose should be included in the initial supervised dataset. Do not invent labels for images that COLMAP did not register. If more frames are registered later, regenerate and validate the pose CSV and data split reproducibly.

---

## 9. Train / Validation / Test Split

Adjacent frames from one video are highly correlated. Randomly splitting individual images could place nearly identical frames in both training and test sets and overstate performance.

The split should therefore be made **by complete video sequence**, not by individual image.

The current proposed assignment is:

| Split | Video sequences |
|---|---|
| Train | `bebop_001_desde_piso`, `bebop_002_avance_bajo`, `bebop_003_elevacion`, `bebop_004_avance_elevacion`, `bebop_005_pared_1`, `bebop_007_avance_giro`, `bebop_009_esquina_opuesta`, `bebop_012_recorrido_completo` |
| Validation | `bebop_008_diagonal`, `bebop_010_recorrido_lateral` |
| Test | `bebop_006_pared_2`, `bebop_011_recorrido_circular` |

The initial expected counts are approximately 410 training images, 77 validation images, and 122 test images, for 609 images total. These counts must be confirmed from the generated CSV; the script should print the actual counts.

The intended output is:

```text
data/poses_colmap_split.csv
```

It should contain the original pose fields plus a `split` column with values `train`, `val`, or `test`.

After generating it, verify the line count, inspect the header and first rows, check actual counts, and confirm that no video identifier occurs in more than one split. The CSV should contain 610 lines if all 609 pose rows are included, plus the header.

A sequence-level split reduces leakage from adjacent frames, but all videos were captured in the same general environment. The evaluation therefore measures generalization to held-out routes within this environment, not generalization to entirely new rooms or environments.

---

## 10. Target Model Architecture

The first model will use ResNet-18 as a visual feature extractor and replace its classification layer with a regression head.

```text
                    Image
                      |
                      v
                  ResNet-18
                      |
                      v
                Visual features
                      |
                      v
                 Regression head
                      |
              +-------+-------+
              |               |
              v               v
           Position        Orientation
            x, y, z       qx, qy, qz, qw
```

The planned output is:

```text
[x, y, z, qx, qy, qz, qw]
```

The output quaternion will be normalized before calculating orientation losses or metrics. The implementation must handle quaternion sign ambiguity because \(q\) and \(-q\) represent the same rotation.

ResNet-18 is the required baseline. ResNet-34 and EfficientNet-B0 are candidates for optional comparison if the baseline is working and the schedule allows.

---

## 11. Planned Training Pipeline

The initial training pipeline will include:

1. Load the split CSV.
2. Read each image and its pose label.
3. Apply consistent image resizing and normalization.
4. Use a PyTorch `Dataset` and `DataLoader`.
5. Initialize ResNet-18 with a seven-value regression head.
6. Normalize position targets using statistics computed from the **training set only**.
7. Normalize predicted quaternions for orientation calculations.
8. Calculate position and orientation losses separately.
9. Train using backpropagation and an optimizer.
10. Monitor training and validation metrics.
11. Save the best model according to a predefined validation metric.
12. Evaluate the selected model on the held-out test set.

The conceptual loss is:

\[
L = L_{position} + \lambda L_{rotation}
\]

Position and rotation losses have different scales and units, so \(\lambda\) must be selected and documented rather than assumed. A suitable rotation loss should respect quaternion sign equivalence or use a rotation-aware formulation.

All preprocessing statistics must be fitted using training data only. The validation set is used for model selection; the test set should not be used repeatedly to tune hyperparameters.

---

## 12. Evaluation

The initial evaluation will measure:

- Position error in the coordinate system of the labels.
- Orientation angular error in degrees.
- Training and validation loss.
- Test-set performance.
- Inference latency.
- Model size and computational cost where relevant.

If the labels remain in COLMAP coordinates, position error is in **COLMAP reconstruction units**, not meters. It must not be reported as meters until a metric-scale alignment has been established.

Orientation error should be computed from rotations, not by naively subtracting Euler angles. For unit quaternions \(q_{pred}\) and \(q_{ref}\), a sign-invariant angular error can be calculated from their relative rotation using the absolute quaternion dot product and clipping numerical values to the valid range.

Because the labels are generated by COLMAP, these metrics quantify agreement with COLMAP estimates. They do not independently establish absolute physical pose accuracy. Independent validation would require an external reference such as surveyed camera positions, Vicon, or another suitably calibrated measurement method.

Results should include the split definition and the number of registered images used from each sequence.

---

## 13. Camera Pose Versus Drone Pose

The current images and COLMAP labels describe camera poses. The camera pose and Bebop body pose are not necessarily identical.

For later drone localization, the system must account for the fixed camera-to-body transformation. Once that transformation and the target reference frame are defined, the intended pipeline is:

```text
Bebop image
    |
    v
ResNet-18
    |
    v
Estimated camera pose
    |
    v
Coordinate-frame alignment
    |
    v
Camera-to-Bebop-body transformation
    |
    v
Pose expressed in the selected ROS reference frame
```

The camera-to-body extrinsic transform must be measured, obtained from reliable device specifications, or calibrated. It must not be silently assumed to be the identity transform.

This project initially targets pose estimation from images. It does not claim autonomous flight control. ROS/Bebop integration is a later test of inference and message/interface compatibility, subject to time and model quality.

---

## 14. Development Environment

### Local machine

- Laptop: HP Pavilion 15-cw1xxx
- CPU: AMD Ryzen 5 3500U
- RAM: 24 GB
- GPU: AMD Radeon Vega 8
- NVIDIA CUDA GPU: not available locally
- OS: Ubuntu 24.04 LTS
- Python: 3.12.3

The laptop is used for data preparation, development, debugging, small experiments, and evaluation. Google Colab or an available NVIDIA GPU may be used for heavier training.

### Main Python libraries

- PyTorch
- torchvision
- NumPy
- OpenCV
- pandas
- matplotlib
- scikit-learn
- tqdm
- Pillow

Dependencies are recorded in `requirements.txt`. For reproducibility, package versions used in the final training run should also be recorded.

---

## 15. Python Virtual Environment

The repository uses a virtual environment named `.venv`, which is intentionally excluded from Git.

Activate it in a new terminal:

```bash
cd ~/drone-6dof-pose-estimation
source .venv/bin/activate
```

Verify Python and the active environment:

```bash
which python
python --version
```

Verify PyTorch and CUDA availability:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On the local laptop, CUDA is expected to be unavailable because there is no NVIDIA CUDA-compatible GPU.

When finished, the environment can be deactivated without deleting it:

```bash
deactivate
```

---

## 16. Initial Environment and Training Verification

The environment has been tested locally and in Google Colab.

### Local verification

- PyTorch and torchvision installed.
- ResNet-18 forward pass with input shape `1 × 3 × 224 × 224` completed.
- Standard ResNet-18 output shape `1 × 1000` verified.
- A modified regression model produced a six-value output for an initial technical test.
- Loss calculation, gradient computation, backpropagation, and weight updates succeeded.
- Local execution used the CPU.

The six-output test was only a technical verification of training operations. It was not the final seven-value pose model and did not use the real pose dataset.

### Google Colab verification

The basic regression/training test also ran in Google Colab with an NVIDIA T4 GPU.

| Check | Laptop | Colab |
|---|---:|---:|
| PyTorch / torchvision | Pass | Pass |
| ResNet-18 forward pass | Pass | Pass |
| Regression output | Pass | Pass |
| Loss calculation | Pass | Pass |
| Gradients and weight update | Pass | Pass |
| CUDA GPU available | No | Yes, NVIDIA T4 |

The installed PyTorch versions differed between environments during these checks. Record the exact environment used for final experiments.

---

## 17. COLMAP Workflow and Quality Checks

The joint COLMAP reconstruction was generated from frames extracted from the twelve Bebop sequences.

The main stages completed so far are:

1. Extract frames from the source videos.
2. Run feature extraction.
3. Run exhaustive matching.
4. Generate the joint reconstruction.
5. Export poses for registered images.
6. Validate image names, numeric fields, and quaternion norms.

The next quality checks should include:

- Number of registered images per source sequence.
- Whether all sequences are connected in the joint reconstruction.
- Whether any sequence has very few registered images.
- Spatial distribution of reconstructed camera centers.
- Camera orientations and whether they are plausible given the recorded movements.
- Reconstruction artifacts, outliers, and possible drift.
- Confirmation of the pose convention implemented by the CSV export script.
- Whether images from different videos were reconstructed in a consistent shared model.

The existence of a pose in the CSV confirms that the image was registered and exported; it does not guarantee that the pose is physically accurate.

Do not rerun feature extraction or exhaustive matching without a specific reason. If a later reconstruction change is necessary, document the configuration and regenerate dependent artifacts consistently.

---

## 18. Repository Structure

The exact structure may evolve as implementation files are added. The main current and planned paths include:

```text
drone-6dof-pose-estimation/
├── data/
│   ├── frames/
│   ├── poses_colmap.csv
│   └── poses_colmap_split.csv       # create or verify
├── results/
│   └── colmap_all_12/               # joint COLMAP outputs
├── src/
│   ├── extract_frames.py
│   └── test_resnet18.py              # technical verification script
├── .gitignore
├── README.md
└── requirements.txt
```

This is a conceptual overview, not a guarantee that every path or file exists exactly as shown. Inspect the repository before relying on a planned path. Large raw videos, COLMAP databases, reconstruction artifacts, virtual environments, and model checkpoints should be handled according to `.gitignore` and the repository's storage policy; they should not be committed blindly.

---

## 19. Experiment Log

For each meaningful experiment, record:

- Date and experiment identifier.
- Objective and hypothesis.
- Dataset version and split.
- Number of images and sequences used.
- Model architecture and pretrained-weight choice.
- Input resolution and preprocessing.
- Position normalization method.
- Quaternion convention and normalization.
- Loss functions and loss weights.
- Optimizer, learning rate, batch size, and number of epochs.
- Training and validation curves.
- Test metrics, reported only after model selection.
- Inference time and hardware.
- Failures, observations, and conclusions.

Save configuration files and random seeds where practical so that results can be reproduced.

---

## 20. Current Status — October 14, 2026

### Project setup

- [x] Git repository created.
- [x] GitHub repository and SSH authentication configured.
- [x] Python virtual environment created.
- [x] Required Python libraries installed.
- [x] Local PyTorch and torchvision verified.
- [x] ResNet-18 forward pass verified.
- [x] Basic regression training operations verified locally.
- [x] Basic regression training operations verified in Google Colab.

### Data collection and COLMAP

- [x] Twelve Bebop 2 videos recorded.
- [x] Frames extracted from all twelve videos.
- [x] COLMAP feature extraction completed.
- [x] COLMAP exhaustive matching completed.
- [x] Joint reconstruction generated.
- [x] Camera poses exported to `data/poses_colmap.csv`.
- [x] Basic pose CSV validation completed: 609 unique images, no missing image files, no invalid numeric rows, and normalized quaternions.
- [ ] Inspect reconstruction coverage and pose plausibility in detail.
- [ ] Verify camera-pose and quaternion conventions in the export code.
- [ ] Create and validate `data/poses_colmap_split.csv`.

### Dataset and model

- [ ] Implement the PyTorch image/pose `Dataset`.
- [ ] Implement `DataLoader` and preprocessing.
- [ ] Define training-only position normalization.
- [ ] Implement a seven-output ResNet-18 regression model.
- [ ] Define and test position and rotation losses.
- [ ] Train the first model on real data.
- [ ] Evaluate on held-out sequences.
- [ ] Save quantitative metrics, plots, and model checkpoints.

### Coordinate alignment and validation

- [ ] Define the physical room coordinate frame.
- [ ] Establish the camera's actual location relative to the chosen origin.
- [ ] Obtain enough reliable physical references to estimate scale, rotation, and translation.
- [ ] Align COLMAP poses to the physical frame if suitable references are available.
- [ ] Investigate independent validation using measured positions or Vicon.
- [ ] Determine the camera-to-Bebop-body transformation before claiming body-pose output.

### Optional work

- [ ] Compare ResNet-34 or EfficientNet-B0 if the baseline is complete.
- [ ] Evaluate a second camera source, such as a webcam, only if time remains.
- [ ] Investigate Gaussian Splatting or synthetic data augmentation only if it addresses a specific need and does not delay the baseline.
- [ ] Test inference with ROS/Bebop if the model and schedule permit.

---

## 21. Immediate Next Steps

The immediate priority is to make the current pose dataset reproducible and technically consistent.

1. Inspect the joint reconstruction and verify registered-image counts by sequence.
2. Inspect the CSV export code to confirm that positions represent camera centers and that the quaternion convention is correct.
3. Create `data/poses_colmap_split.csv` using the proposed sequence-level assignment.
4. Confirm that the split contains all 609 rows and that no sequence is shared between train, validation, and test.
5. Implement a PyTorch `Dataset` that loads an image and its corresponding pose.
6. Test a batch through the loader and confirm image and target tensor shapes.
7. Implement the seven-output ResNet-18 baseline and verify loss and gradient calculations using a small batch.
8. Start real-data training only after the label convention, split, and loss design have been checked.

Do not modify the original `data/poses_colmap.csv` in place. Keep it as the exported COLMAP-label source. Store split assignments and future transformed labels in separate files.

---

## 22. Coordinate Alignment: Decisions Still Required

The intended target frame is the physical room frame:

```text
Origin: floor-level corner where the nets meet
X:      horizontal, into the space along the initial viewing direction
Y:      horizontal, perpendicular to X
Z:      vertical, positive upward
Units:  meters after metric alignment
```

Before transforming the dataset, establish:

- Whether the first camera was exactly at the corner or at a nonzero offset.
- The camera height above the floor.
- Enough reliable physical measurements or position correspondences to estimate a 3D similarity transform.
- Whether a metric reference or Vicon measurement is available.
- The exact pose and quaternion convention in the current CSV.
- The fixed transformation between the camera and the Bebop body, if body pose is required.

Do not simply subtract the first CSV position to define the room origin unless that camera position is known to correspond to the desired physical origin. Do not assume the COLMAP reconstruction scale is metric.

The initial model may be trained in COLMAP coordinates if that enables a useful baseline, but the reported position error must then be expressed in reconstruction units. A separately aligned dataset can be created once the transformation is established.

---

## 23. Model Comparison and Optional Camera Experiment

ResNet-18 is the mandatory baseline. If it trains and evaluates successfully with enough time remaining, candidate models include:

| Model | Position error | Orientation error | Inference time | Model size |
|---|---:|---:|---:|---:|
| ResNet-18 | — | — | — | — |
| ResNet-34 | — | — | — | — |
| EfficientNet-B0 | — | — | — | — |

All comparisons must use the same sequence split, label convention, input preprocessing, and evaluation metrics. Differences in training configuration should be documented.

A second-camera experiment or Gaussian Splatting should not be started before the baseline produces interpretable results. If a webcam experiment is performed, the sequences should cover comparable viewpoints and trajectories. Independent COLMAP reconstructions may have different coordinate frames and scales, so their labels cannot be compared directly without alignment.

---

## 24. ROS and Bebop Integration

The initial objective is to estimate pose from images, not to control the drone.

A later inference test may connect the model output to ROS. Before using the estimate as a drone-body pose, the implementation must:

1. Define the frame represented by the output.
2. Align the output to the desired world or map frame, if metric alignment is available.
3. Convert camera pose to Bebop-body pose using the camera-to-body extrinsic transform.
4. Convert quaternion ordering and coordinate conventions correctly for the ROS message/interface.
5. Measure inference latency and inspect temporal stability.
6. Validate the estimate against an independent reference where possible.

ROS integration is a stretch milestone. A correct offline image-to-pose pipeline and honest evaluation take priority over a rushed flight-control demonstration.

---

## 25. Scope Control

### Required

- Joint COLMAP reconstruction from the twelve Bebop video sequences.
- Export and validation of image-pose labels.
- Sequence-level train/validation/test split.
- PyTorch dataset and loading pipeline.
- ResNet-18 baseline with position and quaternion outputs.
- Quantitative evaluation on held-out sequences.
- Documentation of pose conventions, reconstruction-scale limitations, and evaluation limitations.

### Desirable if time allows

- Physical coordinate-frame alignment and metric scale using reliable references.
- Independent validation using measured positions or Vicon.
- Offline inference test using new Bebop images.
- ROS message/interface integration.
- One additional architecture comparison.

### Optional / only after the baseline

- Webcam dataset comparison.
- Further ResNet-34 or EfficientNet-B0 experiments.
- Gaussian Splatting or synthetic image generation.
- Additional dataset expansion.

The project should not depend on completing every optional technology. A small, reproducible, correctly evaluated baseline is preferable to several unfinished methods.

---

## 26. Final Deliverables

The planned final deliverables are:

1. A documented methodology for generating and checking COLMAP-based pose labels from Bebop images.
2. A reproducible dataset index with sequence-level train/validation/test assignments.
3. A trained ResNet-18 image-to-pose baseline and saved evaluation results.
4. Position and orientation metrics, training curves, and a discussion of limitations.
5. A clear explanation of the difference between COLMAP-estimated labels and independent physical ground truth.
6. A final report and cleaned repository.
7. ROS/Bebop inference integration if it can be completed and validated within the remaining schedule.

The final report must distinguish clearly between results measured in COLMAP reconstruction coordinates and results obtained after any verified metric alignment.
