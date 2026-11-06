# Drone 6-DoF Pose Estimation



Deep-learning-based visual estimation of camera pose from indoor images, using COLMAP-generated pose estimates as initial labels and ResNet-18 as the baseline regression model.



**Project period:** September 1 – November 27, 2026 &#x20;

**\*\*Status updated:**** October 21, 2026



---



## 1. Project Overview



This project investigates whether a convolutional neural network can estimate the 6-DoF pose associated with an indoor image.



A 6-DoF pose describes:



- **\*\*Position:\*\*** \(x, y, z\)

- **\*\*Orientation:\*\*** three rotational degrees of freedom, represented by a unit quaternion \(q_x, q_y, q_z, q_w\)



The planned regression output contains seven numerical values:



```text

[x, y, z, qx, qy, qz, qw]

```



A quaternion uses four values to represent a three-degree-of-freedom rotation. Predicted quaternions must be normalized, and orientation evaluation must account for the fact that \(q\) and \(-q\) represent the same rotation.



The current dataset comes from twelve indoor videos recorded with a Parrot Bebop 2. Frames were extracted and processed together with COLMAP, and the exported camera poses are being used as **\*\*estimated pose labels\*\***.



The initial reproducible image-to-pose pipeline and a 20-epoch ResNet-18 baseline are now implemented. Metric alignment, independent validation, final test evaluation, and the transformation from camera pose to drone-body pose remain to be addressed.



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




The project runs from September 1 to November 27, 2026. The plan below reflects the status on October 21 and prioritizes finishing a reliable baseline before optional extensions.

| Target dates | Milestone | Status |
|---|---|---|
| Sep 1–25 | State-of-the-art review and tool familiarization | Completed / continued as needed |
| Sep 21–Oct 13 | Capture indoor sequences, extract frames, run COLMAP | Completed: twelve videos processed and joint reconstruction generated |
| Oct 13–20 | Inspect labels, implement dataset split and PyTorch data pipeline | Initial CSV validation and video-level split completed; reusable scripts prepared |
| Oct 16–23 | Implement and run the ResNet-18 image-to-pose baseline | Implemented; 20-epoch training run completed |
| Oct 21–30 | Review baseline, evaluate held-out test data, inspect errors | In progress / next priority |
| Oct 30–Nov 6 | Improve baseline and establish quantitative results | Planned; prioritize fixes supported by evaluation |
| Nov 6–13 | Optional architecture comparison or additional data experiment | Optional; only if baseline evaluation is complete |
| Nov 9–20 | Test image-to-pose inference and explore ROS/Bebop integration | Planned; scope depends on baseline quality and time |
| Nov 16–23 | Final validation, plots, metrics, and limitations | Planned |
| Nov 23–27 | Final report, README, repository cleanup, and delivery | Planned |

Dates may overlap where tasks can proceed independently. The first priority is now to evaluate the trained baseline honestly, verify pose conventions and coordinate limitations, and produce reproducible results. Weekends are not included in the main work estimate and provide schedule buffer.

## 4. Development Strategy



The project follows an implementation-first strategy:



1. Review a concept when needed.

2. Implement it in PyTorch.

3. Run a controlled experiment.

4. Inspect predictions, losses, gradients, and metrics.

5. Modify parameters or deliberately test failure cases.

6. Document the outcome.

7. Apply the lesson to the real pose-estimation task.



A separate MNIST exercise is **\*\*not a mandatory milestone\*\***. The local and Colab checks already verified the basic forward pass, regression output, loss computation, gradient calculation, and optimizer update. The next learning steps should use the actual image-pose dataset wherever practical.



---



## 5. Current Methodology




The current pipeline is:

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
  Export camera poses to data/poses_colmap.csv
             |
             v
  Validate image/pose associations and conventions
             |
             v
 Split by complete video into train / val / test
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

The export, validation, and video-split stages now have reusable scripts in `src/`: `export_colmap_poses.py`, `validate_poses.py`, and `split_dataset.py`. The scripts make the current workflow easier to repeat, but the video names and split assignments are specific to this Bebop dataset and should be reviewed when using new data.

The current pose labels come from COLMAP processing the same Bebop images used as model inputs. They are therefore not independent ground-truth measurements.

The project does not use Bebop odometry as the primary label source and does not yet use Vicon measurements. Independent validation remains desirable and must be reported separately from agreement with COLMAP.

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



These values are expressed in the reconstructed coordinate system as exported by the current pose-conversion script. They must **\*\*not\*\*** be assumed to be meters.



The number of registered images varies by sequence. For example, the complete-route sequence has 140 registered images and the circular-route sequence has 94. A sequence with fewer registered images is not automatically invalid, but reconstruction coverage and quality should be inspected.



### 6.3 Important interpretation caveats



- **\*\*Estimated labels:\*\*** COLMAP poses are reconstruction estimates, not independently measured ground truth.

- **\*\*Monocular scale:\*\*** a monocular reconstruction generally has an unknown global scale unless metric information is introduced.

- **\*\*Coordinate frame:\*\*** COLMAP's origin and axes are not automatically aligned with the room or ROS.

- **\*\*Pose convention:\*\*** COLMAP's stored image extrinsics normally represent a world-to-camera transformation. The camera center in the reconstruction frame is computed from the rotation and translation. Verify that the export code makes `x, y, z` and the quaternion consistently describe the intended camera pose.

- **\*\*Camera versus drone:\*\*** the image pose describes the camera. It is not automatically the pose of the Bebop body.

- **\*\*Training target:\*\*** a model trained on these labels learns to reproduce COLMAP's pose estimates and their errors. Good test performance against these labels does not by itself prove accurate real-world localization.



Before final training, document the exact meaning of the exported position and quaternion, quaternion order, rotation direction, and coordinate frame.



---



## 7. Coordinate-Frame Plan



The intended physical frame for future alignment is provisionally defined as follows:



- **\*\*Origin:\*\*** the floor-level corner where the nets meet.

- **\*\*X axis:\*\*** horizontal, pointing into the space in the initial viewing direction of the Bebop.

- **\*\*Y axis:\*\*** horizontal and perpendicular to X, following the other direction of the space.

- **\*\*Z axis:\*\*** vertical and positive upward.

- **\*\*Units:\*\*** meters, after a valid metric-scale alignment.



This is the target convention, not the current COLMAP frame.



The exact physical location of the camera in the first frame has not yet been established. The first image must not automatically be assigned position \((0,0,0)\) merely because recording began near the reference corner.



For positions, the intended similarity transform is:



\[

\mathbf{p}\_{target} = s R_a \mathbf{p}\_{COLMAP} + \mathbf{t}\_a

\]



where \(s\) is the scale factor, \(R_a\) aligns the axes, and \(\mathbf{t}\_a\) establishes the target origin. A consistent transformation must also be applied to orientations.



Potential sources for alignment include measured room dimensions and known camera positions, or independent position references obtained using the INAOE Vicon system. A single known distance can constrain scale but does not determine the complete scale/rotation/translation alignment on its own.



**\*\*Status:\*\*** physical alignment, metric scale, and independent reference measurements are pending. Do not describe the current CSV as metric ground truth.



---



## 8. Dataset and Capture Status




### 8.1 Source videos

Twelve indoor videos were recorded with the Parrot Bebop 2. Frames were extracted from all twelve videos and used for the joint COLMAP workflow. The source sequence names are:

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

The current image folder used by the COLMAP/training workflow is `data/frames_colmap_all/`. If the older `data/frames/` folder also exists, do not assume it is the active dataset without checking the script arguments and actual files.

### 8.2 Current data artifacts

| Artifact / stage | Status |
|---|---|
| Twelve Bebop videos captured | Complete |
| 646 frames extracted | Complete |
| COLMAP feature extraction and exhaustive matching | Complete |
| Joint reconstruction | Complete |
| 609 images registered in the selected model (`sparse/1`) | Complete |
| Poses exported to `data/poses_colmap.csv` | Complete |
| Basic CSV/image/quaternion validation | Complete |
| Video-level split in `data/poses_colmap_split.csv` | Prepared; rerun/check with `src/split_dataset.py` when needed |
| PyTorch dataset and loader | Implemented |
| Seven-output ResNet-18 model and training loop | Implemented |
| 20-epoch training run | Completed |
| Final test evaluation and metric-coordinate alignment | Pending |

### 8.3 Inclusion rule

Only images with a valid exported pose should be included in the initial supervised dataset. Do not invent labels for images that COLMAP did not register. The selected model has 609 registered images from 646 extracted frames; the remaining images should not be assigned fabricated labels. If more frames are registered later, regenerate and validate the pose CSV and split reproducibly.

## 9. Train / Validation / Test Split




Adjacent frames from one video are highly correlated. Randomly splitting individual images could place nearly identical frames in both training and test sets and overstate performance. The split is therefore made by complete video sequence, not by individual image.

The current assignment is defined in `VIDEO_SPLITS` inside `src/split_dataset.py`:

| Split | Video sequences |
|---|---|
| Train | `bebop_001_desde_piso`, `bebop_002_avance_bajo`, `bebop_003_elevacion`, `bebop_004_avance_elevacion`, `bebop_005_pared_1`, `bebop_007_avance_giro`, `bebop_009_esquina_opuesta`, `bebop_012_recorrido_completo` |
| Validation | `bebop_008_diagonal`, `bebop_010_recorrido_lateral` |
| Test | `bebop_006_pared_2`, `bebop_011_recorrido_circular` |

The current registered-image counts are:

| Split | Images |
|---|---:|
| Train | 410 |
| Validation | 77 |
| Test | 122 |
| **Total** | **609** |

The output is `data/poses_colmap_split.csv`, containing the pose fields plus a `split` column with values `train`, `val`, or `test`.

Regenerate the split with:

```bash
python src/split_dataset.py
```

After generating it, inspect the header and counts and confirm that no video identifier occurs in more than one split. The CSV should contain 610 lines if all 609 pose rows are included, plus the header.

A sequence-level split reduces leakage from adjacent frames, but all videos were captured in the same general environment. The evaluation measures generalization to held-out routes within this environment, not generalization to entirely new rooms or environments. Reserve `test` for final evaluation; use `val` for model selection.

## 10. Target Model Architecture



The first model will use ResNet-18 as a visual feature extractor and replace its classification layer with a regression head.



```text

&#x20;                   Image

&#x20;                     |

&#x20;                     v

&#x20;                 ResNet-18

&#x20;                     |

&#x20;                     v

&#x20;               Visual features

&#x20;                     |

&#x20;                     v

&#x20;                Regression head

&#x20;                     |

&#x20;             +-------+-------+

&#x20;             |               |

&#x20;             v               v

&#x20;          Position        Orientation

&#x20;           x, y, z       qx, qy, qz, qw

```



The planned output is:



```text

[x, y, z, qx, qy, qz, qw]

```



The output quaternion will be normalized before calculating orientation losses or metrics. The implementation must handle quaternion sign ambiguity because \(q\) and \(-q\) represent the same rotation.



ResNet-18 is the required baseline. ResNet-34 and EfficientNet-B0 are candidates for optional comparison if the baseline is working and the schedule allows.



---



## 11. Planned Training Pipeline




The initial training pipeline is implemented in `src/train.py` and uses the real image-pose dataset:

1. Load `data/poses_colmap_split.csv`.
2. Read each image and its pose label through the PyTorch dataset.
3. Resize and normalize images.
4. Normalize position targets using statistics computed from the training split only.
5. Normalize predicted quaternions for orientation calculations.
6. Calculate position and orientation losses.
7. Train with backpropagation and an optimizer.
8. Monitor training and validation loss/metrics.
9. Save the best checkpoint based on validation performance.
10. Evaluate the selected checkpoint on the held-out test set (final test evaluation remains pending).

The conceptual loss is:

\[
L = L_{position} + \lambda L_{rotation}
\]

Position and rotation losses have different scales and units, so the implementation and any loss weights should be documented when comparing experiments. The rotation formulation must respect quaternion sign equivalence: `q` and `-q` represent the same rotation.

The current command for a 20-epoch run is:

```bash
python -u -m src.train --epochs 20 --batch-size 8
```

The current implementation uses ResNet-18 pretrained on ImageNet, a seven-value regression output, batch size 8, learning rate `1e-4` by default, and seed 42 by default. Check `python -m src.train --help` for the options supported by the checked-out version.

The best checkpoint and training history are written to `outputs/resnet18/best_model.pth` and `outputs/resnet18/history.json`. All preprocessing statistics must be fitted using training data only. The validation set is used for model selection; the test set must not be repeatedly used to tune hyperparameters.

## 12. Evaluation



The initial evaluation will measure:



- Position error in the coordinate system of the labels.

- Orientation angular error in degrees.

- Training and validation loss.

- Test-set performance.

- Inference latency.

- Model size and computational cost where relevant.



If the labels remain in COLMAP coordinates, position error is in **\*\*COLMAP reconstruction units\*\***, not meters. It must not be reported as meters until a metric-scale alignment has been established.



Orientation error should be computed from rotations, not by naively subtracting Euler angles. For unit quaternions \(q\_{pred}\) and \(q\_{ref}\), a sign-invariant angular error can be calculated from their relative rotation using the absolute quaternion dot product and clipping numerical values to the valid range.



Because the labels are generated by COLMAP, these metrics quantify agreement with COLMAP estimates. They do not independently establish absolute physical pose accuracy. Independent validation would require an external reference such as surveyed camera positions, Vicon, or another suitably calibrated measurement method.



Results should include the split definition and the number of registered images used from each sequence.



---



## 13. Camera Pose Versus Drone Pose



The current images and COLMAP labels describe camera poses. The camera pose and Bebop body pose are not necessarily identical.



For later drone localization, the system must account for the fixed camera-to-body transformation. Once that transformation and the target reference frame are defined, the intended pipeline is:



```text

Bebop image

&#x20;   |

&#x20;   v

ResNet-18

&#x20;   |

&#x20;   v

Estimated camera pose

&#x20;   |

&#x20;   v

Coordinate-frame alignment

&#x20;   |

&#x20;   v

Camera-to-Bebop-body transformation

&#x20;   |

&#x20;   v

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

python -c "import torch; print('PyTorch:', torch.\_\_version\_\_); print('CUDA available:', torch.cuda.is_available())"

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




The repository structure has evolved since the initial setup. The main paths relevant to the current pipeline are:

```text
drone-6dof-pose-estimation/
├── data/
│   ├── frames_colmap_all/
│   ├── poses_colmap.csv
│   └── poses_colmap_split.csv
├── results/
│   └── colmap_all_12/
│       └── sparse/1_txt/images.txt
├── src/
│   ├── extract_frames.py
│   ├── export_colmap_poses.py
│   ├── validate_poses.py
│   ├── split_dataset.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── outputs/
│   └── resnet18/
│       ├── best_model.pth
│       └── history.json
├── .gitignore
├── README.md
└── requirements.txt
```

This is the expected structure for the current workflow, not a guarantee that every generated artifact is tracked in Git. Inspect the repository before relying on a path. Large raw videos, image folders, COLMAP databases/reconstruction outputs, virtual environments, and model checkpoints should be handled according to `.gitignore` and the repository's storage policy; do not commit them blindly.

### Reusable data-preparation scripts

- `src/export_colmap_poses.py`: reads the selected COLMAP text model and exports camera centers and quaternions to a CSV.
- `src/validate_poses.py`: checks required columns, image associations, finite values, duplicates, and quaternion norms.
- `src/split_dataset.py`: creates a train/validation/test CSV using the current video-level assignment.
- `src/dataset.py`: loads and preprocesses images and pose labels.
- `src/model.py`: defines the ResNet-18 regression model and pose-loss components.
- `src/train.py`: trains, validates, and saves the best checkpoint.
- `src/extract_frames.py`: extracts frames from source videos.

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



## 20. Current Status — October 21, 2026




**Status updated: October 21, 2026**

### Project setup

- [x] Git repository and GitHub SSH access configured.
- [x] Python virtual environment and core libraries installed.
- [x] Local PyTorch/torchvision and basic regression operations verified.
- [x] Basic training workflow verified in Google Colab.

### Data collection and COLMAP

- [x] Twelve Bebop 2 videos recorded and frames extracted.
- [x] COLMAP feature extraction, exhaustive matching, and joint reconstruction completed.
- [x] Poses exported to `data/poses_colmap.csv`.
- [x] Basic validation completed: 609 unique registered images, no missing image files in the checked dataset, no invalid numeric rows, and normalized quaternions.
- [x] Video-level split established: 410 train, 77 validation, 122 test.
- [x] Reusable export, validation, and split scripts prepared.
- [ ] Continue inspecting reconstruction coverage and pose plausibility by sequence.
- [ ] Reconfirm export conventions before any coordinate transformation.

### Dataset and model

- [x] PyTorch image/pose dataset and preprocessing implemented.
- [x] Training-only position normalization implemented.
- [x] Seven-output ResNet-18 regression model implemented.
- [x] Position/orientation loss and validation loop implemented.
- [x] 20-epoch training run completed.
- [x] Best checkpoint saved at `outputs/resnet18/best_model.pth`; best epoch 16, validation loss `0.09146746872113896`.
- [ ] Run and document the final held-out test evaluation.
- [ ] Inspect predicted poses and error cases; save plots and quantitative metrics.
- [ ] Measure inference latency and model size.

### Coordinate alignment and independent validation

- [ ] Define and establish the physical room coordinate frame.
- [ ] Establish the camera's actual location relative to the chosen origin.
- [ ] Obtain enough reliable physical references to estimate scale, rotation, and translation.
- [ ] Align COLMAP poses to the physical frame if suitable references are available.
- [ ] Investigate independent validation using measured positions or Vicon.
- [ ] Determine the camera-to-Bebop-body transformation before claiming body-pose output.

### Integration and optional work

- [ ] Test offline inference on new images.
- [ ] Explore ROS/Bebop inference integration if baseline quality and time permit.
- [ ] Compare ResNet-34 or EfficientNet-B0 only after baseline evaluation.
- [ ] Consider a second camera source or Gaussian Splatting only if it addresses a specific need and does not delay required deliverables.

## 21. Immediate Next Steps




The immediate priority is to evaluate and understand the first real-data baseline before expanding the project scope.

1. Confirm the current checkpoint, history file, dataset CSV, and split counts.
2. Run the held-out test evaluation without using test results to tune the model.
3. Report position error in COLMAP reconstruction units until metric scale is established, and report orientation angular error in degrees.
4. Inspect per-sequence errors and representative predictions.
5. Verify the camera-center and quaternion conventions in the export script before transforming labels.
6. Decide how to use a known physical distance and room axes to align the reconstruction; one distance alone constrains scale but does not determine the full 3D alignment.
7. Measure inference latency and record model size/hardware.
8. Attempt ROS/Bebop integration only after offline inference is reproducible and the output frame is clearly defined.

Do not modify `data/poses_colmap.csv` in place. Keep it as the exported COLMAP-label source. Store split assignments and any future aligned/transformed labels in separate files.

### Reproducible data and training commands

Run from the repository root with `.venv` activated:

```bash
python src/export_colmap_poses.py
python src/validate_poses.py
python src/split_dataset.py
python -u -m src.train --epochs 20 --batch-size 8
```

The first three steps need to be rerun when the COLMAP model, image set, or split configuration changes. Training can be rerun without regenerating the CSVs if those inputs have not changed. Confirm script arguments with `--help` where supported.

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

1. Documented methodology for generating and checking COLMAP-based pose labels from Bebop images.
2. Reproducible dataset index with video-level train/validation/test assignments.
3. Trained ResNet-18 image-to-pose baseline and saved training history/checkpoint.
4. Quantitative held-out test results, position/orientation metrics, and training/validation plots.
5. Discussion of COLMAP pose conventions, unknown monocular scale, and the difference between estimated labels and independent physical ground truth.
6. A clear account of generalization limits and failure cases.
7. Final report and cleaned repository.
8. ROS/Bebop inference integration if it can be completed and validated within the remaining schedule.

The final report must distinguish between results measured in COLMAP reconstruction coordinates and results obtained after any verified metric alignment. A low validation loss alone is not sufficient evidence of accurate physical localization.



| Archivo | Función | ¿Se utiliza ahora? |
|---|---|---|
| `extract_frames.py` | Extrae imágenes de los videos. | No, si ya tienes los frames actualizados. |
| `validate_poses.py` | Comprueba la consistencia de las poses y los datos de referencia, según su implementación. | Sí, cuando preparemos las nuevas poses o coordenadas. |
| `export_colmap_poses.py` | Exporta las poses reconstruidas por COLMAP a un CSV. | Después de la reconstrucción. |
| `align_colmap_poses.py` | Alinea las coordenadas de COLMAP con el sistema de referencia físico y aplica la transformación y escala definidas en el código. | Después de exportar las poses, adaptándolo a las nuevas referencias. |
| `split_dataset.py` | Divide las imágenes y las poses en conjuntos de entrenamiento, validación y prueba. | No para ejecutar COLMAP; sí posteriormente para entrenar la red neuronal. |
| `dataset.py` | Prepara los datos para entrenar o evaluar el modelo. | No durante la reconstrucción. |
| `model.py` | Define la arquitectura de la red neuronal. | No durante la reconstrucción. |
| `train.py` | Entrena la red neuronal. | Después de preparar y validar el nuevo dataset. |
| `test_resnet18.py` | Evalúa el modelo ResNet-18, según su implementación. | Después del entrenamiento. |