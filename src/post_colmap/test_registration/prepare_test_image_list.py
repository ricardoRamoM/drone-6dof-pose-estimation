
#!/usr/bin/env python3

from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[3]

IMAGE_DIR = ROOT / "dataset/v2_1/frames_all"
DATABASE = ROOT / "results_colmap/v2_1_registration/database.db"
OUTPUT = ROOT / "results_colmap/v2_1_registration/test_images.txt"

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    names = [p.name for p in images]

    if not images:
        raise RuntimeError("No se encontraron imágenes de test.")

    if len(names) != len(set(names)):
        raise RuntimeError("Hay nombres de imágenes duplicados.")

    for path in images:
        if not path.is_file():
            raise RuntimeError(f"Imagen inexistente: {path}")

    connection = sqlite3.connect(
        f"file:{DATABASE}?mode=ro", uri=True
    )

    try:
        existing_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM images"
            )
        }
    finally:
        connection.close()

    collisions = sorted(set(names) & existing_names)

    if collisions:
        raise RuntimeError(
            f"Se encontraron {len(collisions)} nombres duplicados "
            f"en la base de datos: {collisions[:10]}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(names) + "\n")

    print("=" * 60)
    print("LISTA DE IMÁGENES DE TEST")
    print("=" * 60)
    print(f"Imágenes encontradas: {len(images)}")
    print(f"Colisiones con DB: {len(collisions)}")
    print(f"Archivo: {OUTPUT}")
    print(f"Primera imagen: {names[0]}")
    print(f"Última imagen: {names[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
