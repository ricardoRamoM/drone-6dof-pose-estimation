
from pathlib import Path
from collections import Counter
import re

MODEL_DIR = Path("results_colmap/v2_1/sparse/0_txt")
IMAGES_TXT = MODEL_DIR / "images.txt"

if not IMAGES_TXT.exists():
    raise FileNotFoundError(f"No existe: {IMAGES_TXT}")

groups = Counter()
excluded_1er_intento = []
other = []
total = 0

for line in IMAGES_TXT.read_text().splitlines():
    fields = line.split(maxsplit=9)

    # Las líneas de imagen contienen ID, quaternion, translation,
    # camera_id y nombre/ruta de imagen.
    if len(fields) != 10 or not fields[0].isdigit():
        continue

    image_path = fields[9]

    # Evitar confundir las líneas de observaciones 2D con imágenes.
    if not re.search(r"\.jpe?g$", image_path, re.IGNORECASE):
        continue

    total += 1
    basename = Path(image_path).name

    # La ruta completa es importante: no basta con el nombre del archivo.
    if "1er Intento" in image_path:
        groups["1er Intento (EXCLUIR)"] += 1
        excluded_1er_intento.append(image_path)

    elif re.match(r"bebop_test_\d+_frame_\d+\.jpg$", basename):
        match = re.match(r"(bebop_test_\d+)_frame_", basename)
        groups[match.group(1)] += 1

    elif re.match(r"ref_frame\d+\.jpg$", basename):
        groups["references"] += 1

    elif re.match(r"bebop_00[1-5]_(?:val|train)_frame_\d+\.jpg$", basename):
        match = re.match(r"(bebop_00[1-5]_(?:val|train))_frame_", basename)
        groups[match.group(1)] += 1

    else:
        groups["OTHER"] += 1
        other.append(image_path)

print(f"\nTotal de imágenes: {total}")

print("\nDistribución:")
for group, count in sorted(groups.items()):
    print(f"{group}: {count}")

print(f"\nImágenes que se excluirían: {len(excluded_1er_intento)}")
print("\nEjemplos de rutas de 1er Intento:")
for path in excluded_1er_intento[:15]:
    print(path)

print(f"\nImágenes OTHER: {len(other)}")
for path in other[:30]:
    print(path)

print("\nComprobación:")
print(f"Total contabilizado: {sum(groups.values())}")
