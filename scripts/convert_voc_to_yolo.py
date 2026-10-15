from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data"

SOURCES = [
    RAW_DIR / "China_Drone" / "train",
    RAW_DIR / "China_MotorBike" / "train",
]

CLASS_MAP = {
    "D00": 0,  # Longitudinal crack
    "D10": 1,  # Transverse crack
    "D20": 2,  # Alligator crack
    "D40": 3,  # Pothole
}


def convert_box(width, height, xmin, ymin, xmax, ymax):
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return x_center, y_center, box_width, box_height


def main():
    images_output = OUTPUT_DIR / "images"
    labels_output = OUTPUT_DIR / "labels"

    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_objects = 0
    ignored_objects = 0

    for source in SOURCES:
        images_dir = source / "images"
        xml_dir = source / "annotations" / "xmls"

        for xml_path in xml_dir.glob("*.xml"):
            root = ET.parse(xml_path).getroot()

            filename = root.findtext("filename")
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            image_path = images_dir / filename

            if not image_path.exists():
                print(f"Imagen no encontrada: {image_path}")
                continue

            yolo_lines = []

            for obj in root.findall("object"):
                class_name = obj.findtext("name")

                if class_name not in CLASS_MAP:
                    ignored_objects += 1
                    continue

                class_id = CLASS_MAP[class_name]
                box = obj.find("bndbox")

                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                x, y, w, h = convert_box(
                    width, height, xmin, ymin, xmax, ymax
                )

                yolo_lines.append(
                    f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                )

                total_objects += 1

            if not yolo_lines:
                continue

            shutil.copy2(image_path, images_output / filename)

            label_path = labels_output / f"{Path(filename).stem}.txt"
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

            total_images += 1

    print("\nConversión terminada")
    print(f"Imágenes procesadas: {total_images}")
    print(f"Objetos convertidos: {total_objects}")
    print(f"Objetos ignorados: {ignored_objects}")


if __name__ == "__main__":
    main()