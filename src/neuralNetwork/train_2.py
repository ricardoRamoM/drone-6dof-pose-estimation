# train_2.py
# Entrenamiento de ResNetPoseRegressor para regresión de pose 6-DoF.
# Incluye: ReduceLROnPlateau, early stopping, logging de LR,
# weight decay (L2), exportación de predicciones de val por muestra.

"""
Uso:
# Prueba corta (2 épocas):
# python src/neuralNetwork/train_2.py --epochs 2 --seed 42 --run-tag smoke_test

# Entrenamiento completo (E13, 100 épocas, scheduler + early stopping):
# python src/neuralNetwork/train_2.py --epochs 100 --batch-size 8 --lr 1e-4 \
#     --seed 42 --scheduler-patience 10 --early-stopping-patience 50 \
#     --run-tag scheduler_ep100

# E14 (regularización L2, weight decay = 1e-4):
# python src/neuralNetwork/train_2.py --epochs 120 --batch-size 8 --lr 1e-4 \
#     --seed 42 --weight-decay 1e-4 --early-stopping-patience 50 \
#     --run-tag e14_wd
"""

from pathlib import Path
import argparse
import csv
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

DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_SEED = 42
DEFAULT_EARLY_STOPPING_PATIENCE = 50
DEFAULT_WEIGHT_DECAY = 0.0  # 0.0 = sin regularización (comportamiento previo)


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
    """Crea una carpeta única por ejecución."""
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_number = _next_run_number(BASE_OUTPUT_DIR, args.version)

    lr_str = f"{args.lr:g}".replace(".", "p")
    name = (
        f"{args.version}_run_{run_number:02d}"
        f"_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{lr_str}"
        f"_sd{args.seed}"
    )
    # Solo añadimos wd al nombre si es distinto de 0,
    # para no romper la nomenclatura de ejecuciones anteriores.
    if args.weight_decay > 0:
        wd_str = f"{args.weight_decay:g}".replace(".", "p")
        name += f"_wd{wd_str}"
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
    """Evalúa el modelo y devuelve métricas medias del loader."""
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
        "position_error_mean_cm": float(np.mean(position_errors) * 100),
        "position_error_median_cm": float(np.median(position_errors) * 100),
        "orientation_error_mean_deg": float(np.mean(angular_errors)),
        "orientation_error_median_deg": float(np.median(angular_errors)),
    }


def export_val_predictions(
    model,
    checkpoint_path: Path,
    val_loader,
    device,
    position_mean,
    position_std,
    output_path: Path,
) -> None:
    """Carga el mejor checkpoint y guarda predicciones por muestra en CSV."""
    if not checkpoint_path.exists():
        print("No hay checkpoint para exportar predicciones de val.")
        return

    # weights_only=False para compatibilidad con PyTorch >= 2.6
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    sample_idx = 0

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            pred_pos = predictions[:, :3] * position_std + position_mean
            tgt_pos = targets[:, :3]

            pos_err = torch.linalg.vector_norm(pred_pos - tgt_pos, dim=1)

            pred_q = normalize_quaternion(predictions[:, 3:7])
            tgt_q = normalize_quaternion(targets[:, 3:7])
            dot = torch.sum(pred_q * tgt_q, dim=1).abs().clamp(0.0, 1.0)
            ang_err = 2.0 * torch.acos(dot) * (180.0 / np.pi)

            for i in range(images.shape[0]):
                rows.append({
                    "sample_index": sample_idx,
                    "target_x": float(tgt_pos[i, 0]),
                    "target_y": float(tgt_pos[i, 1]),
                    "target_z": float(tgt_pos[i, 2]),
                    "pred_x": float(pred_pos[i, 0]),
                    "pred_y": float(pred_pos[i, 1]),
                    "pred_z": float(pred_pos[i, 2]),
                    "pos_error_cm": float(pos_err[i]) * 100,
                    "target_qw": float(tgt_q[i, 0]),
                    "target_qx": float(tgt_q[i, 1]),
                    "target_qy": float(tgt_q[i, 2]),
                    "target_qz": float(tgt_q[i, 3]),
                    "pred_qw": float(pred_q[i, 0]),
                    "pred_qx": float(pred_q[i, 1]),
                    "pred_qy": float(pred_q[i, 2]),
                    "pred_qz": float(pred_q[i, 3]),
                    "ang_error_deg": float(ang_err[i]),
                })
                sample_idx += 1

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predicciones de val exportadas: {output_path}  ({len(rows)} muestras)")


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
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="Épocas sin mejora antes de reducir LR (default: 10).",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Factor multiplicativo del LR al reducirse (default: 0.5).",
    )
    parser.add_argument(
        "--scheduler-threshold",
        type=float,
        default=1e-3,
        help=(
            "Mejora mínima para considerar 'mejora' (default: 1e-3). "
            "Bájalo a 1e-4 si el scheduler baja el LR demasiado pronto."
        ),
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="LR mínimo que el scheduler puede alcanzar (default: 1e-6).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "Épocas sin mejora de val_loss antes de detener el entrenamiento. "
            f"0 = desactivado (default: {DEFAULT_EARLY_STOPPING_PATIENCE})."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=(
            "Coeficiente de regularización L2 (weight decay) del optimizador. "
            f"0.0 = desactivado (default: {DEFAULT_WEIGHT_DECAY}). "
            "Para E14 usar 1e-4."
        ),
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs y batch-size deben ser positivos.")
    if args.weight_decay < 0:
        raise ValueError("weight-decay debe ser >= 0.")

    set_seed(args.seed)
    training_start_time = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PyTorch:", torch.__version__)
    print("Dispositivo:", device)
    print("Versión dataset:", args.version)
    print("Épocas:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Seed:", args.seed)
    print("Scheduler patience:", args.scheduler_patience)
    print("Scheduler factor:", args.scheduler_factor)
    print("Scheduler threshold:", args.scheduler_threshold)
    print("Scheduler min_lr:", args.min_lr)
    print("Early stopping patience:", args.early_stopping_patience)
    print("Weight decay:", args.weight_decay)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        min_lr=args.min_lr,
    )

    best_val_loss = float("inf")
    history = []

    epochs_without_improvement = 0
    early_stopped = False
    stopped_epoch = None

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

        # LR con el que se entrenó esta época (antes de scheduler.step)
        lr_this_epoch = optimizer.param_groups[0]["lr"]

        scheduler.step(val_metrics["loss"])
        lr_next_epoch = optimizer.param_groups[0]["lr"]

        if lr_next_epoch < lr_this_epoch:
            print(
                f"  Scheduler: LR reducido "
                f"{lr_this_epoch:.6g} -> {lr_next_epoch:.6g}"
            )

        epoch_duration = time.perf_counter() - epoch_start_time
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],   # <-- añadir esta línea
            "epoch_duration_sec": epoch_duration,
            "learning_rate": lr_this_epoch,
            **val_metrics,
        }
        history.append(record)

        # Diferencia train/val: útil para diagnosticar sobreajuste.
        generalization_gap = val_metrics["loss"] - train_loss

        print(
            f"\nÉpoca {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"gap={generalization_gap:+.5f} | "
            f"lr={lr_this_epoch:.6g} | "
            f"tiempo={format_duration(epoch_duration)}"
        )
        print(
            f"  Error de posición: "
            f"media={val_metrics['position_error_mean']*100:.2f} cm "
            f"({val_metrics['position_error_mean']:.4f} m), "
            f"mediana={val_metrics['position_error_median']*100:.2f} cm "
            f"({val_metrics['position_error_median']:.4f} m)"
        )
        print(
            f"  Error angular: "
            f"media={val_metrics['orientation_error_mean_deg']:.3f}°, "
            f"mediana={val_metrics['orientation_error_median_deg']:.3f}°"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_epoch_time = time.perf_counter() - training_start_time

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
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
        else:
            epochs_without_improvement += 1

        with open(history_path, "w") as file:
            json.dump(history, file, indent=2)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"\nEarly stopping activado: "
                f"{epochs_without_improvement} épocas sin mejorar "
                f"val_loss (mejor={best_val_loss:.5f}, "
                f"actual={val_metrics['loss']:.5f})."
            )
            early_stopped = True
            stopped_epoch = epoch
            break

    total_duration = time.perf_counter() - training_start_time
    epochs_completed = len(history)

    update_run_info(
        run_dir,
        end_time=datetime.now().isoformat(timespec="seconds"),
        total_duration_sec=total_duration,
        best_val_loss=best_val_loss,
        epochs_completed=epochs_completed,
        early_stopped=early_stopped,
        stopped_epoch=stopped_epoch,
        final_learning_rate=optimizer.param_groups[0]["lr"],
    )

    print("\nEntrenamiento terminado.")
    print("Versión dataset:", args.version)
    print(f"Épocas configuradas: {args.epochs}")
    print(f"Épocas completadas: {epochs_completed}")
    if early_stopped:
        print(f"Early stopping en época {stopped_epoch}")
    print("Batch size:", args.batch_size)
    print("Learning rate inicial:", args.lr)
    print("Learning rate final:", optimizer.param_groups[0]["lr"])
    print("Weight decay:", args.weight_decay)
    print("Semilla:", args.seed)
    print(f"Tiempo total de entrenamiento: {format_duration(total_duration)}")
    print(
        f"Tiempo medio por época: "
        f"{format_duration(total_duration / epochs_completed)}"
    )
    print("Mejor pérdida de validación:", best_val_loss)
    print("Checkpoint:", checkpoint_path)
    print("Historial:", history_path)

    val_predictions_path = run_dir / "val_predictions.csv"
    export_val_predictions(
        model=model,
        checkpoint_path=checkpoint_path,
        val_loader=val_loader,
        device=device,
        position_mean=position_mean,
        position_std=position_std,
        output_path=val_predictions_path,
    )

    print(
        "\nNota: no se ha evaluado el conjunto test. "
        "Se reservará para la evaluación final."
    )


if __name__ == "__main__":
    main()