#!/usr/bin/env python3
"""
Transform COLMAP camera poses to a first metric, room-aligned coordinate frame.

Stage 1:
- Use A (COLMAP 3D point ID 170) as the provisional origin.
- Map COLMAP axes to the desired room axes:
      X_room =  X_colmap
      Y_room =  Z_colmap
      Z_room = -Y_colmap
  This is a proper rotation (determinant +1), equivalent to Rx(-90 deg)
  under the standard active, column-vector convention.
- Estimate scale from H (COLMAP point ID 69549), assumed 1.5 m above
  the floor, using the mean COLMAP Y-coordinate of floor points A, B, C.
- Transform positions and camera-to-world quaternions.
- Preserve the input CSV; write a new CSV and a JSON record of the transform.

IMPORTANT:
This assumes the input CSV quaternion columns qx,qy,qz,qw encode the
camera-to-world rotation, as produced by the project's COLMAP exporter.
If they instead encode world-to-camera rotations, the orientation step must
be adapted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# Run from the repository root:
INPUT_CSV = Path("data/poses_colmap.csv")
OUTPUT_CSV = Path("data/poses_colmap_metric_aligned_A.csv")
TRANSFORM_JSON = Path("data/colmap_alignment_stage1.json")

# COLMAP sparse point coordinates supplied by the user.
# Point A is the provisional origin.
A = np.array([-2.515356062806398,  2.0644037617079811, 2.7209781183111477])
B = np.array([ 0.5881692641482713, 1.9552731873089813, 2.6263317515954294])
C = np.array([-2.3861302102872437, 2.0303224460517790, 0.23574600008142579])
H = np.array([ 4.5187524282573168, -1.7481500450259917, 6.4166542357986254])

D = np.array([2.15598, 1.95524, 2.64580])
E = np.array([2.14671, 1.93701, 4.35668])
F = np.array([0.58817, 1.95527, 2.62633])

HEIGHT_H_METERS = 1.5

# Transformación de coordenadas COLMAP a mundo FLU.
#
# Sistema de destino:
#   +X = adelante (Forward)
#   +Y = izquierda (Left)
#   +Z = arriba (Up)
#
# Mapeo:
#   X_FLU =  Z_COLMAP
#   Y_FLU = -X_COLMAP
#   Z_FLU = -Y_COLMAP
#
# D -> E debe apuntar aproximadamente hacia +X.
# D -> F debe apuntar aproximadamente hacia +Y.
#
# La matriz es una rotación propia: det(M) = +1.

M = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
])


def transform_position(
    point_colmap: np.ndarray,
    origin_colmap: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Transform a COLMAP position to the provisional metric FLU frame."""
    return scale * (rotation_matrix @ (point_colmap - origin_colmap))


def validate_reference_points(
    D: np.ndarray,
    E: np.ndarray,
    F: np.ndarray,
    origin: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float,
) -> None:
    """Check the FLU directions and metric distances of D, E and F."""

    D_flu = transform_position(D, origin, rotation_matrix, scale)
    E_flu = transform_position(E, origin, rotation_matrix, scale)
    F_flu = transform_position(F, origin, rotation_matrix, scale)

    de = E_flu - D_flu
    df = F_flu - D_flu

    print("\n=== Validación de referencias físicas ===")
    print(f"D -> E: {de}")
    print(f"D -> F: {df}")

    print(f"Distancia D -> E: {np.linalg.norm(de):.4f} m")
    print(f"Distancia D -> F: {np.linalg.norm(df):.4f} m")

    if de[0] <= 0:
        raise ValueError("D -> E no apunta hacia +X (adelante).")

    if df[1] <= 0:
        raise ValueError("D -> F no apunta hacia +Y (izquierda).")

    print("Dirección D -> E: X positivo, verificada.")
    print("Dirección D -> F: Y positivo, verificada.")

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"No encuentro {INPUT_CSV}. Ejecuta este script desde la raíz "
            "del repositorio y verifica la ruta del CSV."
        )

    df = pd.read_csv(INPUT_CSV)

    required = {"x", "y", "z", "qx", "qy", "qz", "qw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas {sorted(missing)} en {INPUT_CSV}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    if not np.allclose(M @ M.T, np.eye(3), atol=1e-8):
        raise RuntimeError("La matriz de alineación no es ortogonal.")
    det_m = float(np.linalg.det(M))
    if not np.isclose(det_m, 1.0, atol=1e-8):
        raise RuntimeError(f"La matriz debe ser una rotación propia; det={det_m}")

    # Because the user's chosen mapping makes COLMAP -Y point upward,
    # estimate floor level using the mean Y coordinate of floor points A, B, C.
    # Escala provisional usando H, situado a 1.5 m sobre el piso.
    # El eje vertical FLU es -Y_COLMAP.

    floor_y_colmap = float(np.mean([A[1], B[1], C[1]]))
    height_separation_colmap = floor_y_colmap - float(H[1])

    if not np.isfinite(height_separation_colmap) or height_separation_colmap <= 0:
        raise ValueError(
            "La separación vertical de H respecto al piso debe ser positiva. "
            "Revisa las coordenadas de A, B, C y H."
        )

    scale_m_per_colmap_unit = (
        HEIGHT_H_METERS / height_separation_colmap
    )

    validate_reference_points(
        D=D,
        E=E,
        F=F,
        origin=A,
        rotation_matrix=M,
        scale=scale_m_per_colmap_unit,
    )

    print("\n=== Escala métrica provisional ===")
    print(f"Altura física de H: {HEIGHT_H_METERS:.3f} m")
    print(
        "Separación COLMAP entre H y el piso: "
        f"{height_separation_colmap:.6f}"
    )
    print(
        "Escala estimada: "
        f"{scale_m_per_colmap_unit:.6f} m/unidad COLMAP"
    )



    # Positions: subtract provisional origin A, rotate axes, then scale.
    xyz_colmap = df[["x", "y", "z"]].to_numpy(dtype=float)
    xyz_aligned_m = (M @ (xyz_colmap - A).T).T * scale_m_per_colmap_unit

    out = df.copy()
    out[["x", "y", "z"]] = xyz_aligned_m

    # Orientation: assumes q columns are camera-to-world in COLMAP coordinates.
    # Convert each old camera-to-world rotation to the aligned world frame:
    # R_camera_to_room = M @ R_camera_to_colmap.
    quats_xyzw = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    norms = np.linalg.norm(quats_xyzw, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("Hay cuaterniones inválidos o de norma cero.")
    quats_xyzw = quats_xyzw / norms[:, None]
    rotations_colmap = Rotation.from_quat(quats_xyzw)
    rotations_room = Rotation.from_matrix(M @ rotations_colmap.as_matrix())
    quats_room_xyzw = rotations_room.as_quat()
    # q and -q represent the same rotation; keep output normalized.
    quats_room_xyzw /= np.linalg.norm(quats_room_xyzw, axis=1)[:, None]
    out[["qx", "qy", "qz", "qw"]] = quats_room_xyzw

    quat_norms = np.linalg.norm(
        out[["qx", "qy", "qz", "qw"]].to_numpy(),
        axis=1
    )

    print("\n=== Validación de cuaterniones ===")
    print(f"Norma mínima: {quat_norms.min():.8f}")
    print(f"Norma máxima: {quat_norms.max():.8f}")
    print(f"Norma media:  {quat_norms.mean():.8f}")
    print(f"¿Todos son unitarios?: {np.allclose(quat_norms, 1.0, atol=1e-6)}")

    # RPY angles from the TRANSFORMED camera-to-world quaternions.
    # SciPy lowercase "xyz" uses extrinsic XYZ rotations, equivalent to
    # the common roll-pitch-yaw representation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    # Angles are expressed in degrees; near gimbal lock, RPY can be unstable.
    rpy_deg = rotations_room.as_euler("xyz", degrees=True)
    out["roll_deg"] = rpy_deg[:, 0]
    out["pitch_deg"] = rpy_deg[:, 1]
    out["yaw_deg"] = rpy_deg[:, 2]

    # Comprobar hacia dónde apunta el eje óptico de la cámara.
    # En el sistema óptico, el eje Z es la dirección de visión.
    R_room = rotations_room.as_matrix()
    camera_forward = R_room[:, :, 2]

    print("\n=== Dirección del eje óptico de la cámara ===")
    print("Primeras 5 imágenes:")
    for i in range(min(5, len(camera_forward))):
        fx, fy, fz = camera_forward[i]
        print(
            f"{out.iloc[i]['image_name']}: "
            f"X={fx:+.3f}, Y={fy:+.3f}, Z={fz:+.3f}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # Diagnostics using the same transform, to document how measured
    # distances compare with COLMAP and the height-based scale.
    def dist(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    d_ab = dist(A, B)
    d_de_colmap = dist(
        np.array([2.15598, 1.95524, 2.6458]),
        np.array([2.14671, 1.93701, 4.35668]),
    )
    d_df_colmap = dist(
        np.array([2.15598, 1.95524, 2.6458]),
        B,
    )
    summary = {
        "input_csv": str(INPUT_CSV),
        "output_csv": str(OUTPUT_CSV),
        "provisional_origin": "A (COLMAP point ID 170)",
        "point_A_colmap": A.tolist(),
        "point_B_colmap": B.tolist(),
        "point_C_colmap": C.tolist(),
        "point_H_colmap": H.tolist(),
        "H_height_m": HEIGHT_H_METERS,
        "floor_y_colmap_mean_ABC": floor_y_colmap,
        "H_vertical_separation_colmap_units_using_axis_mapping": height_separation_colmap,
        "scale_m_per_colmap_unit": scale_m_per_colmap_unit,
        "axis_mapping": {
            "x_room": "z_colmap",
            "y_room": "-x_colmap",
            "z_room": "-y_colmap",
            "convention": "FLU (Forward-Left-Up)",
        },
        "rotation_matrix_colmap_to_room": M.tolist(),
        "rotation_determinant": det_m,
        "diagnostic_distances": {
            "AB_colmap_units": d_ab,
            "AB_after_scale_m": d_ab * scale_m_per_colmap_unit,
            "AB_physical_claim_m": 1.22,
            "DE_colmap_units": d_de_colmap,
            "DE_after_scale_m": d_de_colmap * scale_m_per_colmap_unit,
            "DE_physical_claim_m": 0.64,
            "DF_colmap_units": d_df_colmap,
            "DF_after_scale_m": d_df_colmap * scale_m_per_colmap_unit,
            "DF_physical_claim_m": 0.60,
        },
        "quaternion_assumption": "qx,qy,qz,qw encode camera-to-world orientation",
        "euler_angles": {
            "columns": ["roll_deg", "pitch_deg", "yaw_deg"],
            "convention": "extrinsic XYZ / common RPY, degrees",
            "source": "transformed camera-to-world quaternions",
        },
        "notes": [
            "World axes: FLU (Forward-Left-Up).",
            "X points forward, Y points left, Z points upward.",
            "A is the provisional origin; the final room-corner origin "
            "has not been applied.",
            "Scale is estimated from H at 1.5 m above the floor.",
            "Distances DE=0.64 m and DF=0.60 m are validation references.",
            "Camera orientations must not be interpreted as drone-body "
            "orientations without an additional camera-to-body transform.",
        ],
    }

    TRANSFORM_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"CSV original: {INPUT_CSV}")
    print(f"CSV nuevo:   {OUTPUT_CSV}")
    print(f"Transformación registrada en: {TRANSFORM_JSON}")
    print()
    print(f"Y medio del piso (A,B,C): {floor_y_colmap:.6f} unidades COLMAP")
    print(f"Separación vertical H-piso: {height_separation_colmap:.6f} unidades COLMAP")
    print(f"Escala provisional: {scale_m_per_colmap_unit:.6f} m/unidad COLMAP")
    print()
    print(f"A-B: {d_ab:.4f} unidades -> {d_ab * scale_m_per_colmap_unit:.3f} m (medida: 1.220 m)")
    print(f"D-E: {d_de_colmap:.4f} unidades -> {d_de_colmap * scale_m_per_colmap_unit:.3f} m (medida: 0.640 m)")
    print(f"D-F: {d_df_colmap:.4f} unidades -> {d_df_colmap * scale_m_per_colmap_unit:.3f} m (medida: 0.600 m)")
    print()
    print("Columnas RPY añadidas: roll_deg, pitch_deg, yaw_deg.")
    print("Convención: extrínseca XYZ (RPY común), en grados.")
    print(f"CSV de salida contiene {len(out)} filas y estas columnas RPY: "
          f"{[c for c in out.columns if c.endswith('_deg')]}")
    print()
    print("IMPORTANTE: revisa que las orientaciones de entrada sean cámara-a-mundo.")
    print("No se modificaron los archivos originales.")


if __name__ == "__main__":
    main()
