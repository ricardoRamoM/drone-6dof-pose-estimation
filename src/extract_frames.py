from pathlib import Path
import argparse
import cv2

# Ejemplo de uso para cinco imagenes por segundo:
# python src/extract_frames.py --fps 5

def extract_frames(video_path, output_dir, fps, start_seconds, end_seconds):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir el video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        print(f"ERROR: No se pudo leer la información de {video_path}")
        cap.release()
        return

    duration = total_frames / video_fps

    if start_seconds + end_seconds >= duration:
        print(f"ERROR: Los recortes son demasiado grandes para {video_path.name}")
        cap.release()
        return

    # Convertir los tiempos a índices de frames originales.
    start_frame = round(start_seconds * video_fps)
    end_frame = total_frames - round(end_seconds * video_fps)

    # Intervalo de muestreo para obtener aproximadamente los FPS deseados.
    interval = max(1, round(video_fps / fps))

    # Limpiar JPG anteriores de esta carpeta para evitar duplicados
    # o imágenes sobrantes de una ejecución anterior.
    for old_image in output_dir.glob("*.jpg"):
        old_image.unlink()

    print(f"\nVideo: {video_path.name}")
    print(f"FPS originales: {video_fps:.2f}")
    print(f"Duración original: {duration:.2f} s")
    print(f"Inicio excluido: {start_seconds:.2f} s")
    print(f"Final excluido: {end_seconds:.2f} s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    saved_count = 0

    while frame_index < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        if (frame_index - start_frame) % interval == 0:
            filename = output_dir / f"frame_{saved_count:05d}.jpg"

            success = cv2.imwrite(
                str(filename),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if success:
                saved_count += 1
            else:
                print(f"ERROR al guardar: {filename}")

        frame_index += 1

    cap.release()

    print(f"Frames extraídos: {saved_count}")
    print(f"Directorio: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae frames de videos para el dataset de pose 6-DoF."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/videos"),
        help="Carpeta con los videos originales."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/frames"),
        help="Carpeta de salida para los frames."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0, # dos imagenes por segundo
        help="Cantidad aproximada de imágenes extraídas por segundo."
    )

    parser.add_argument(
        "--start-seconds",
        type=float,
        default=5.0,
        help="Segundos que se descartan al inicio de cada video."
    )

    parser.add_argument(
        "--end-seconds",
        type=float,
        default=2.5,
        help="Segundos que se descartan al final de cada video."
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps debe ser mayor que cero.")

    if args.start_seconds < 0 or args.end_seconds < 0:
        parser.error("Los tiempos de recorte no pueden ser negativos.")

    videos = sorted(args.input.glob("*.avi"))

    if not videos:
        print(f"No se encontraron videos AVI en {args.input}")
        return

    for video in videos:
        output_dir = args.output / video.stem

        extract_frames(
            video_path=video,
            output_dir=output_dir,
            fps=args.fps,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds
        )


if __name__ == "__main__":
    main()