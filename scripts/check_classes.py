from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

sources = [
    ROOT / "data/raw/China_Drone/train/annotations/xmls",
    ROOT / "data/raw/China_MotorBike/train/annotations/xmls",
]

VALID_CLASSES = {"D00", "D10", "D20", "D40"}

classes = Counter()

total_images = 0
valid_images = 0
empty_after_filter = 0

for source in sources:
    for xml_path in source.glob("*.xml"):
        total_images += 1

        root = ET.parse(xml_path).getroot()
        image_classes = []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            classes[name] += 1
            image_classes.append(name)

        if any(name in VALID_CLASSES for name in image_classes):
            valid_images += 1
        else:
            empty_after_filter += 1

print("\nClases encontradas:")
for name, count in sorted(classes.items()):
    print(f"{name}: {count}")

print("\nImágenes:")
print(f"Total: {total_images}")
print(f"Con D00/D10/D20/D40: {valid_images}")
print(f"Sin clases válidas: {empty_after_filter}")