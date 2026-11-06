#!/usr/bin/env python3
# extract_frames.py
# Extrae frames de videos para el dataset de pose 6-DoF.

"""Extrae frames de videos Bebop para el dataset de pose 6-DoF.

Genera (por defecto, con --version v2_0):
  dataset/v2_0/frames/<video_stem>/   frame_00000.jpg, frame_00001.jpg, ...
  dataset/v2_0/frames_all/            <video_stem>_frame_00000.jpg, ...

Lee los videos AVI desde (por defecto):
  dataset/v2_0/videos/

El índice empieza en cero para cada video y cuenta solo las imágenes extraídas,
no los frames originales del AVI.

Ejemplos:
  python src/pre_colmap/extract_frames.py 
  python src/pre_colmap/extract_frames.py --version v2_0 --clean-output SOLO PARA EXISTENTES
  python src/pre_colmap/extract_frames.py --version v2_0 --cut-mode global --start-seconds 5 --end-seconds 5 --clean-output
  python src/pre_colmap/extract_frames.py --version v2_1 --fps 3 --clean-output
"""

from pathlib import Path
import argparse
import cv2


# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------

# Versión del dataset usada si no se pasa --version por línea de comandos.
# Cambia este valor para mover todas las salidas a otra versión.
DEFAULT_DATASET_VERSION = "v2_0"

# Raíz del dataset. Bajo ella se crea <version>/videos, <version>/frames y
# <version>/frames_all.
DATASET_ROOT = Path("dataset")

# Valores por defecto de extracción (sobreescribibles por CLI).
DEFAULT_FPS = 2.0
DEFAULT_CUT_MODE = "global"  # "global", "per-video"
DEFAULT_START_SECONDS = 5.0
DEFAULT_END_SECONDS = 2.5

# Segundos excluidos al inicio y al final de cada video (modo per-video).
VIDEO_CUTS = {
    "bebop_001_val.avi": {"start_seconds": 5, "end_seconds": 5},
    "bebop_002_train.avi": {"start_seconds": 0, "end_seconds": 2},
    "bebop_003_train.avi": {"start_seconds": 0, "end_seconds": 2},
    "bebop_004_train.avi": {"start_seconds": 5, "end_seconds": 7},
    "bebop_005_train.avi": {"start_seconds": 5, "end_seconds": 6},
}


def get_cuts(video_path: Path, cut_mode: str, start_seconds: float,
             end_seconds: float) -> tuple[float, float]:
    """Selecciona recortes globales o los definidos para un video."""
    if cut_mode == "global":
        return start_seconds, end_seconds

    cuts = VIDEO_CUTS.get(video_path.name, VIDEO_CUTS.get(video_path.stem))
    if cuts is None:
        raise ValueError(
            f"No hay recortes definidos en VIDEO_CUTS para {video_path.name}."
        )
    start = cuts.get("start_seconds", 0)
    end = cuts.get("end_seconds", 0)
    for key, value in (("start_seconds", start), ("end_seconds", end)):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{video_path.name}: {key} debe ser >= 0.")
    return float(start), float(end)


def extract_frames(video_path: Path, per_video_dir: Path, colmap_dir: Path,
                   fps: float, start_seconds: float, end_seconds: float,
                   jpeg_quality: int = 95) -> int:
    """Extrae imágenes con índices consecutivos en ambas ubicaciones."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"No se pudo leer FPS/número de frames de {video_path.name}"
        )

    duration = total_frames / video_fps
    if start_seconds + end_seconds >= duration:
        cap.release()
        raise ValueError(
            f"Los recortes suman {start_seconds + end_seconds:.2f}s y "
            f"la duración de {video_path.name} es {duration:.2f}s."
        )

    start_frame = min(total_frames - 1, int(round(start_seconds * video_fps)))
    end_frame = max(
        start_frame + 1,
        total_frames - int(round(end_seconds * video_fps))
    )
    interval = max(1, int(round(video_fps / fps)))

    per_video_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVideo: {video_path.name}")
    print(f"  Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS originales: {video_fps:.3f}; duración: {duration:.2f}s")
    print(f"  Recorte: inicio {start_seconds:.2f}s; final {end_seconds:.2f}s")
    print(f"  Muestreo: cada {interval} frames (~{video_fps / interval:.3f} FPS)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    extracted_idx = 0
    saved_count = 0

    while frame_index < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_index - start_frame) % interval == 0:
            filename_individual = f"frame_{extracted_idx:05d}.jpg"
            filename_colmap = f"{video_path.stem}_frame_{extracted_idx:05d}.jpg"
            per_video_path = per_video_dir / filename_individual
            colmap_path = colmap_dir / filename_colmap

            params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ok_video = cv2.imwrite(str(per_video_path), frame, params)
            ok_colmap = cv2.imwrite(str(colmap_path), frame, params)

            if ok_video and ok_colmap:
                saved_count += 1
                extracted_idx += 1
            else:
                for path in (per_video_path, colmap_path):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                print(f"ERROR al guardar el frame: {filename_individual}")

        frame_index += 1

    cap.release()
    print(f"  Frames guardados: {saved_count}")
    print(f"  Carpeta individual: {per_video_dir}")
    print(f"  Copia para COLMAP: {colmap_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extrae frames de videos para el dataset de pose 6-DoF."
    )
    parser.add_argument("--version", default=DEFAULT_DATASET_VERSION,
                        help=("Versión del dataset vX_X. Determina las rutas por defecto: "
                              "dataset/<version>/videos, dataset/<version>/frames "
                              f"y dataset/<version>/frames_all "
                              f"(por defecto: {DEFAULT_DATASET_VERSION})."))
    parser.add_argument("--input", type=Path, default=None,
                        help=("Carpeta con los videos AVI. "
                              "Si no se indica, se usa dataset/<version>/videos."))
    parser.add_argument("--output", type=Path, default=None,
                        help=("Carpeta raíz para las carpetas individuales por video. "
                              "Si no se indica, se usa dataset/<version>/frames."))
    parser.add_argument("--colmap-output", type=Path, default=None,
                        help=("Carpeta plana con todos los frames para COLMAP. "
                              "Si no se indica, se usa dataset/<version>/frames_all."))
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"Frecuencia aproximada de extracción en frames por segundo "
                             f"(por defecto: {DEFAULT_FPS}).")
    parser.add_argument("--cut-mode", choices=("global", "per-video"),
                        default=DEFAULT_CUT_MODE,
                        help=(f"global: mismos recortes para todos; per-video: usa VIDEO_CUTS "
                              f"(por defecto: {DEFAULT_CUT_MODE})."))
    parser.add_argument("--start-seconds", type=float, default=DEFAULT_START_SECONDS,
                        help=f"Segundos excluidos al inicio en modo global "
                             f"(por defecto: {DEFAULT_START_SECONDS}).")
    parser.add_argument("--end-seconds", type=float, default=DEFAULT_END_SECONDS,
                        help=f"Segundos excluidos al final en modo global "
                             f"(por defecto: {DEFAULT_END_SECONDS}).")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="Calidad JPEG de 0 a 100 (por defecto 95).")
    parser.add_argument("--clean-output", action="store_true",
                        help="Borra solo JPG dentro de las carpetas de salida antes de extraer.")
    args = parser.parse_args()

    # Validaciones básicas de argumentos.
    if args.fps <= 0:
        parser.error("--fps debe ser mayor que cero.")
    if args.start_seconds < 0 or args.end_seconds < 0:
        parser.error("Los tiempos de recorte no pueden ser negativos.")
    if not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality debe estar entre 0 y 100.")

    # La versión no debe contener separadores de ruta para evitar sorpresas.
    if not args.version or "/" in args.version or "\\" in args.version:
        parser.error("--version no puede estar vacío ni contener separadores de ruta.")

    # Resolver rutas en función de la versión (si no se sobreescriben por CLI).
    dataset_version_dir = DATASET_ROOT / args.version
    if args.input is None:
        args.input = dataset_version_dir / "videos"
    if args.output is None:
        args.output = dataset_version_dir / "frames"
    if args.colmap_output is None:
        args.colmap_output = dataset_version_dir / "frames_all"

    # Ahora sí podemos validar que la carpeta de entrada exista.
    if not args.input.is_dir():
        parser.error(f"No existe la carpeta de entrada: {args.input}")

    videos = sorted(args.input.glob("*.avi"))
    if not videos:
        parser.error(f"No se encontraron videos AVI en {args.input}")

    if args.clean_output:
        # Solo elimina JPG de las carpetas de salida; no toca los AVI.
        for root in (args.output, args.colmap_output):
            if root.exists():
                for image_path in root.rglob("*.jpg"):
                    image_path.unlink()

    total_saved = 0
    errors = []
    for video in videos:
        try:
            start_seconds, end_seconds = get_cuts(
                video, args.cut_mode, args.start_seconds, args.end_seconds
            )
            total_saved += extract_frames(
                video_path=video,
                per_video_dir=args.output / video.stem,
                colmap_dir=args.colmap_output,
                fps=args.fps,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                jpeg_quality=args.jpeg_quality,
            )
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            print(f"ERROR: {exc}")

    print("\n===== RESUMEN =====")
    print(f"Versión del dataset: {args.version}")
    print(f"Entrada (videos):    {args.input}")
    print(f"Videos encontrados:  {len(videos)}")
    print(f"Frames guardados (copias incluidas en ambas ubicaciones): {total_saved}")
    print(f"Carpetas individuales: {args.output}")
    print(f"Frames para COLMAP:    {args.colmap_output}")
    if errors:
        print(f"Videos con errores: {len(errors)}")
        for error in errors:
            print(f" - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()