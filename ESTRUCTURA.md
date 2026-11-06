# Estructura del Proyecto
```text
.
├── dataset
│   ├── v1_0
│   │   ├── frames
│   │   │   ├── bebop_001_desde_piso  [31 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_002_avance_bajo  [37 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_003_elevacion  [32 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_004_avance_elevacion  [36 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_005_pared_1  [30 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_006_pared_2  [28 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_007_avance_giro  [52 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_008_diagonal  [30 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_009_esquina_opuesta  [57 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_010_recorrido_lateral  [47 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_011_recorrido_circular  [94 entries exceeds filelimit, not opening dir]
│   │   │   └── bebop_012_recorrido_completo  [172 entries exceeds filelimit, not opening dir]
│   │   ├── frames_all  [646 entries exceeds filelimit, not opening dir]
│   │   ├── videos
│   │   │   ├── bebop_001_desde_piso.avi
│   │   │   ├── bebop_002_avance_bajo.avi
│   │   │   ├── bebop_003_elevacion.avi
│   │   │   ├── bebop_004_avance_elevacion.avi
│   │   │   ├── bebop_005_pared_1.avi
│   │   │   ├── bebop_006_pared_2.avi
│   │   │   ├── bebop_007_avance_giro.avi
│   │   │   ├── bebop_008_diagonal.avi
│   │   │   ├── bebop_009_esquina_opuesta.avi
│   │   │   ├── bebop_010_recorrido_lateral.avi
│   │   │   ├── bebop_011_recorrido_circular.avi
│   │   │   └── bebop_012_recorrido_completo.avi
│   │   ├── colmap_alignment_stage1.json
│   │   ├── poses_colmap.csv
│   │   ├── poses_colmap_metric_aligned_A.csv
│   │   └── poses_colmap_split.csv
│   ├── v1_0_v2_0
│   │   ├── frames
│   │   │   ├── bebop_001_desde_piso  [31 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_001_val  [325 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_002_avance_bajo  [37 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_002_train  [165 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_003_elevacion  [32 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_003_train  [222 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_004_avance_elevacion  [36 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_004_train  [268 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_005_pared_1  [30 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_005_train  [319 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_006_pared_2  [28 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_007_avance_giro  [52 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_008_diagonal  [30 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_009_esquina_opuesta  [57 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_010_recorrido_lateral  [47 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_011_recorrido_circular  [94 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_012_recorrido_completo  [172 entries exceeds filelimit, not opening dir]
│   │   │   └── ref_frame
│   │   │       ├── ref_frame001.jpg
│   │   │       ├── ref_frame002.jpg
│   │   │       ├── ref_frame003.jpg
│   │   │       ├── ref_frame004.jpg
│   │   │       ├── ref_frame005.jpg
│   │   │       ├── ref_frame006.jpg
│   │   │       ├── ref_frame007.jpg
│   │   │       ├── ref_frame008.jpg
│   │   │       ├── ref_frame009.jpg
│   │   │       ├── ref_frame010.jpg
│   │   │       ├── ref_frame011.jpg
│   │   │       ├── ref_frame012.jpg
│   │   │       ├── ref_frame013.jpg
│   │   │       ├── ref_frame014.jpg
│   │   │       └── ref_frame015.jpg
│   │   └── frames_all  [1315 entries exceeds filelimit, not opening dir]
│   ├── v2_0
│   │   ├── 1
│   │   ├── bebop_tests
│   │   │   ├── bebop_001_val.avi
│   │   │   ├── bebop_002_train.avi
│   │   │   ├── bebop_003_train.avi
│   │   │   ├── bebop_004_train.avi
│   │   │   ├── bebop_005_train.avi
│   │   │   ├── bebop_test_001.avi
│   │   │   ├── bebop_test_002.avi
│   │   │   ├── bebop_test_003.avi
│   │   │   ├── bebop_test_004.avi
│   │   │   ├── bebop_test_005.avi
│   │   │   ├── bebop_test_006.avi
│   │   │   ├── bebop_test_007.avi
│   │   │   ├── bebop_test_008.avi
│   │   │   ├── bebop_test_009.avi
│   │   │   ├── bebop_test_010.avi
│   │   │   └── bebop_test_011.avi
│   │   ├── frames
│   │   │   ├── bebop_001_val  [325 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_002_train  [165 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_003_train  [222 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_004_train  [268 entries exceeds filelimit, not opening dir]
│   │   │   ├── bebop_005_train  [319 entries exceeds filelimit, not opening dir]
│   │   │   └── ref
│   │   │       ├── ref_frame001.jpg
│   │   │       ├── ref_frame002.jpg
│   │   │       ├── ref_frame003.jpg
│   │   │       ├── ref_frame004.jpg
│   │   │       ├── ref_frame005.jpg
│   │   │       ├── ref_frame006.jpg
│   │   │       ├── ref_frame007.jpg
│   │   │       ├── ref_frame008.jpg
│   │   │       ├── ref_frame009.jpg
│   │   │       ├── ref_frame010.jpg
│   │   │       ├── ref_frame011.jpg
│   │   │       ├── ref_frame012.jpg
│   │   │       ├── ref_frame013.jpg
│   │   │       ├── ref_frame014.jpg
│   │   │       └── ref_frame015.jpg
│   │   ├── frames_all  [1314 entries exceeds filelimit, not opening dir]
│   │   ├── videos
│   │   │   └── bebop_test_001.avi
│   │   ├── colmap_reference_positions.txt
│   │   └── poses_split.csv
│   ├── .~lock.poses_colmap_metric_aligned_A.csv#
│   └── .~lock.poses_colmap_split.csv#
├── outputs_nn
│   ├── resnet18
│   │   ├── v2_0_run_00_ep10_bs4_lr0p0001
│   │   │   ├── best_model.pth
│   │   │   └── history.json
│   │   ├── v2_0_run_01_ep2_bs2_lr0p0001
│   │   │   ├── best_model.pth
│   │   │   ├── history.json
│   │   │   └── run_info.json
│   │   └── v2_0_run_02_ep4_bs1_lr0p0001
│   │       ├── best_model.pth
│   │       ├── history.json
│   │       └── run_info.json
│   └── resnet18_v1.0
│       ├── best_model_2epochs.pth
│       ├── best_model.pth
│       ├── history_2epochs.json
│       └── history.json
├── results_colmap
│   ├── test_012
│   │   ├── sparse
│   │   │   └── 0
│   │   │       ├── cameras.bin
│   │   │       ├── images.bin
│   │   │       ├── points3D.bin
│   │   │       └── project.ini
│   │   └── database.db
│   ├── v1_0
│   │   ├── sparse
│   │   │   ├── 0
│   │   │   │   ├── cameras.bin
│   │   │   │   ├── images.bin
│   │   │   │   ├── points3D.bin
│   │   │   │   └── project.ini
│   │   │   ├── 0_txt
│   │   │   │   ├── cameras.txt
│   │   │   │   ├── images.txt
│   │   │   │   └── points3D.txt
│   │   │   ├── 1
│   │   │   │   ├── cameras.bin
│   │   │   │   ├── images.bin
│   │   │   │   ├── points3D.bin
│   │   │   │   └── project.ini
│   │   │   └── 1_txt
│   │   │       ├── cameras.txt
│   │   │       ├── images.txt
│   │   │       └── points3D.txt
│   │   ├── database.db
│   │   ├── database.db-shm
│   │   ├── database.db-wal
│   │   ├── no_registradas_modelo_principal.txt
│   │   ├── no_registradas_ningun_modelo.txt
│   │   ├── registradas_cualquier_modelo.txt
│   │   ├── registradas_modelo_0.txt
│   │   ├── registradas.txt
│   │   └── todas.txt
│   ├── v1_0_v2_0
│   │   ├── sparse
│   │   │   ├── 0
│   │   │   │   ├── cameras.bin
│   │   │   │   ├── images.bin
│   │   │   │   ├── points3D.bin
│   │   │   │   └── project.ini
│   │   │   ├── 1
│   │   │   │   ├── cameras.bin
│   │   │   │   ├── images.bin
│   │   │   │   ├── points3D.bin
│   │   │   │   └── project.ini
│   │   │   └── 1_txt
│   │   │       ├── cameras.txt
│   │   │       ├── images.txt
│   │   │       └── points3D.txt
│   │   ├── database.db
│   │   └── imagenes_restantes.txt
│   └── v2_0
│       ├── alignment
│       │   ├── alignment_summary.txt
│       │   ├── manual_review.txt
│       │   ├── poses_drone_FLU.csv
│       │   ├── pose_validation_all.csv
│       │   ├── pose_validation_summary.txt
│       │   └── similarity_transform_colmap_to_FLU.txt
│       ├── sparse
│       │   ├── 0
│       │   │   ├── cameras.bin
│       │   │   ├── images.bin
│       │   │   ├── points3D.bin
│       │   │   └── project.ini
│       │   └── 0_txt
│       │       ├── cameras.txt
│       │       ├── images.txt
│       │       └── points3D.txt
│       ├── database.db
│       └── filtering_report.txt
├── src
│   ├── neuralNetwork
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── test_resnet18.py
│   │   └── train.py
│   ├── post_colmap
│   │   ├── filter
│   │   │   └── filter_colmap_reconstruction.py
│   │   ├── v1
│   │   │   ├── align_colmap_poses.py
│   │   │   ├── export_colmap_poses.py
│   │   │   ├── split_dataset.py
│   │   │   └── validate_poses.py
│   │   ├── align_colmap_to_flu.py
│   │   ├── split_dataset.py
│   │   └── validate_poses.py
│   └── pre_colmap
│       └── extract_frames.py
├── .venv
│   ├── bin  [22 entries exceeds filelimit, not opening dir]
│   ├── include
│   │   └── python3.12
│   ├── lib
│   │   └── python3.12
│   │       └── site-packages  [99 entries exceeds filelimit, not opening dir]
│   ├── lib64 -> lib
│   ├── share
│   │   └── man
│   │       └── man1
│   │           ├── isympy.1
│   │           └── ttx.1
│   └── pyvenv.cfg
├── .gitignore
├── ESTRUCTURA.md
├── README copy.md
├── README.md
├── README_proyecto_drone_6dof.md
└── requirements.txt

94 directories, 156 files
```
