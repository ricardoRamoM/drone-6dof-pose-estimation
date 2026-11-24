
from pathlib import Path
from collections import Counter
from PIL import Image

IMAGE_DIR = Path("dataset/v2_1/frames_all")

extensions = {".jpg", ".jpeg", ".png"}

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

print("=" * 60)
print("INSPECCIÓN DE IMÁGENES DE TEST")
print("=" * 60)

print(f"Directorio: {IMAGE_DIR}")
print(f"Total de imágenes: {len(images)}")

dimensions = Counter()
errors = []

for path in images:
    try:
        with Image.open(path) as img:
            dimensions[img.size] += 1
    except Exception as exc:
        errors.append((path.name, str(exc)))

print("\nDistribución de dimensiones:")
for size, count in dimensions.most_common():
    print(f"  {size[0]}x{size[1]}: {count} imágenes")

print("\nPrimeras 10 imágenes:")
for path in images[:10]:
    try:
        with Image.open(path) as img:
            print(f"  {path.name}: {img.size}")
    except Exception:
        print(f"  {path.name}: ERROR")

print(f"\nImágenes con errores: {len(errors)}")
for name, error in errors[:10]:
    print(f"  {name}: {error}")

print("\n" + "=" * 60)
print("FIN DE LA INSPECCIÓN")
print("=" * 60)
