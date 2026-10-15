from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "data" / "images"
LABELS_DIR = ROOT / "data" / "labels"
OUTPUT_DIR = ROOT / "data" / "dataset"

SEED = 42
random.seed(SEED)


def main():
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    random.shuffle(images)

    total = len(images)

    train_end = int(total * 0.80)
    val_end = train_end + int(total * 0.10)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in splits.items():

        image_output = OUTPUT_DIR / "images" / split_name
        label_output = OUTPUT_DIR / "labels" / split_name

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:

            label_path = LABELS_DIR / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"Etiqueta no encontrada: {label_path}")
                continue

            shutil.copy2(image_path, image_output / image_path.name)
            shutil.copy2(label_path, label_output / label_path.name)

    print("\nDivisión terminada")
    print(f"Total: {total}")

    for name, files in splits.items():
        percentage = len(files) / total * 100
        print(f"{name}: {len(files)} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()