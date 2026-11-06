# train.py
# Entrenamiento de ResNetPoseRegressor para regresión de pose 6-DoF.

"""
Uso:
# python src/neuralNetwork/train.py 
# python src/neuralNetwork/train.py --epochs 20 --batch-size 8 --lr 1e-4 --version v2_0
"""

from pathlib import Path
import argparse
import json
import random
import re
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ColmapPoseDataset, DEFAULT_DATASET_VERSION
from model import ResNetPoseRegressor, normalize_quaternion, pose_loss


ROOT = Path(__file__).resolve().parents[2]
BASE_OUTPUT_DIR = ROOT / "outputs_nn" / "resnet18"

DEFAULT_EPOCHS = 4 #20
DEFAULT_BATCH_SIZE = 1 #8
DEFAULT_LR = 1e-4 
DEFAULT_SEED = 42

# learning rate (tasa de aprendizaje) es un hiperparámetro que controla 
# la magnitud de los pasos de actualización de los pesos durante el entrenamiento.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def format_duration(seconds: float) -> str:
    """Convierte segundos a un string legible tipo '1h 23m 45.6s'."""
    return str(timedelta(seconds=round(seconds, 1)))


def _next_run_number(base_dir: Path, version: str) -> int:
    """Siguiente número de ejecución libre para una versión dada."""
    if not base_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(version)}_run_(\d+)_")
    numbers = [
        int(m.group(1))
        for p in base_dir.iterdir()
        if p.is_dir() and (m := pattern.match(p.name))
    ]
    return max(numbers) + 1 if numbers else 1


def create_run_directory(args) -> Path:
    """Crea una carpeta única por ejecución.

    Ejemplo: outputs_nn/resnet18/v2_0_run_03_ep10_bs4_lr0p0001/
    """
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_number = _next_run_number(BASE_OUTPUT_DIR, args.version)

    lr_str = f"{args.lr:g}".replace(".", "p")
    name = (
        f"{args.version}_run_{run_number:02d}"
        f"_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{lr_str}"
    )
    if args.run_tag:
        name += f"_{args.run_tag}"

    run_dir = BASE_OUTPUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def update_run_info(run_dir: Path, **extra) -> None:
    """Añade/actualiza campos en run_info.json (lo crea si no existe)."""
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update(extra)
    info_path.write_text(json.dumps(info, indent=2))


def evaluate(model, loader, device, position_mean, position_std):
    model.eval()

    total_loss = 0.0
    total_position_loss = 0.0
    total_orientation_loss = 0.0
    total_images = 0

    position_errors = []
    angular_errors = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            loss, pos_loss, ori_loss = pose_loss(
                predictions,
                targets,
                position_mean=position_mean,
                position_std=position_std,
            )

            batch_size = images.shape[0]

            total_loss += loss.item() * batch_size
            total_position_loss += pos_loss.item() * batch_size
            total_orientation_loss += ori_loss.item() * batch_size
            total_images += batch_size

            # Desnormalizar para evaluar en coordenadas originales.
            predicted_position = (
                predictions[:, :3] * position_std + position_mean
            )

            position_error = torch.linalg.vector_norm(
                predicted_position - targets[:, :3], dim=1,
            )

            predicted_quaternion = normalize_quaternion(predictions[:, 3:7])
            target_quaternion = normalize_quaternion(targets[:, 3:7])

            dot = torch.sum(
                predicted_quaternion * target_quaternion, dim=1
            ).abs().clamp(0.0, 1.0)

            angular_error = 2.0 * torch.acos(dot) * (180.0 / np.pi)

            position_errors.extend(position_error.cpu().tolist())
            angular_errors.extend(angular_error.cpu().tolist())

    return {
        "loss": total_loss / total_images,
        "position_loss": total_position_loss / total_images,
        "orientation_loss": total_orientation_loss / total_images,
        "position_error_mean": float(np.mean(position_errors)),
        "position_error_median": float(np.median(position_errors)),
        "orientation_error_mean_deg": float(np.mean(angular_errors)),
        "orientation_error_median_deg": float(np.median(angular_errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--version",
        default=DEFAULT_DATASET_VERSION,
        help=(
            "Versión del dataset bajo dataset/ "
            f"(default: {DEFAULT_DATASET_VERSION})."
        ),
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="No descargar ni utilizar pesos preentrenados.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Texto opcional añadido al nombre de la carpeta de la ejecución.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs y batch-size deben ser positivos.")

    set_seed(args.seed)
    training_start_time = time.perf_counter()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PyTorch:", torch.__version__)
    print("Dispositivo:", device)
    print("Versión dataset:", args.version)
    print("Épocas:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)

    run_dir = create_run_directory(args)
    checkpoint_path = run_dir / "best_model.pth"
    history_path = run_dir / "history.json"

    print("Carpeta de ejecución:", run_dir)

    update_run_info(
        run_dir,
        args=vars(args),
        device=str(device),
        torch_version=torch.__version__,
        start_time=datetime.now().isoformat(timespec="seconds"),
    )    

    train_dataset = ColmapPoseDataset(split="train", version=args.version)
    val_dataset = ColmapPoseDataset(split="val", version=args.version)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # Estadísticas calculadas exclusivamente con el conjunto train.
    train_positions = torch.tensor(
        train_dataset.df[["x", "y", "z"]].to_numpy(),
        dtype=torch.float32,
    )

    position_mean = train_positions.mean(dim=0).to(device)
    position_std = train_positions.std(
        dim=0, unbiased=False
    ).clamp_min(1e-6).to(device)

    print("\nMedia de posición:", position_mean.cpu().tolist())
    print("Desviación estándar:", position_std.cpu().tolist())

    model = ResNetPoseRegressor(
        pretrained=not args.no_pretrained
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    history = []

    print(
        f"\n=== Inicio del entrenamiento: "
        f"{datetime.now().isoformat(sep=' ', timespec='seconds')} ==="
    )
    print("=" * 60)

    update_run_info(
        run_dir,
        training_start_time=datetime.now().isoformat(timespec="seconds"),
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()

        running_loss = 0.0
        seen = 0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(images)

            loss, _, _ = pose_loss(
                predictions,
                targets,
                position_mean=position_mean,
                position_std=position_std,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "La pérdida dejó de ser finita. "
                    "Se detiene el entrenamiento."
                )

            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / seen

        val_metrics = evaluate(
            model, val_loader, device, position_mean, position_std,
        )

        epoch_duration = time.perf_counter() - epoch_start_time
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_duration_sec": epoch_duration,
            **val_metrics,
        }
        history.append(record)

        print(
            f"\nÉpoca {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"tiempo={format_duration(epoch_duration)}"
        )
        print(
            f"  Error de posición: "
            f"media={val_metrics['position_error_mean']:.5f}, "
            f"mediana={val_metrics['position_error_median']:.5f} "
            f"(unidades COLMAP)"
        )
        print(
            f"  Error angular: "
            f"media={val_metrics['orientation_error_mean_deg']:.3f}°, "
            f"mediana={val_metrics['orientation_error_median_deg']:.3f}°"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch_time = time.perf_counter() - training_start_time

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "position_mean": position_mean.cpu(),
                    "position_std": position_std.cpu(),
                    "best_val_loss": best_val_loss,
                    "dataset_version": args.version,
                    "training_time_sec": best_epoch_time,
                    "run_dir": str(run_dir),
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print("  Mejor modelo guardado:", checkpoint_path)

        with open(history_path, "w") as file:    
            json.dump(history, file, indent=2)

    total_duration = time.perf_counter() - training_start_time

    update_run_info(
        run_dir,
        end_time=datetime.now().isoformat(timespec="seconds"),
        total_duration_sec=total_duration,
        best_val_loss=best_val_loss,
    ) 

    print("\nEntrenamiento terminado.")
    print("Versión dataset:", args.version)
    print("Épocas:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print(f"Tiempo total de entrenamiento: {format_duration(total_duration)}")
    print(
        f"Tiempo medio por época: "
        f"{format_duration(total_duration / args.epochs)}"
    )
    print("Mejor pérdida de validación:", best_val_loss)
    print("Checkpoint:", checkpoint_path)
    print("Historial:", history_path)
    print(
        "\nNota: no se ha evaluado el conjunto test. "
        "Se reservará para la evaluación final."
    )


if __name__ == "__main__":
    main()