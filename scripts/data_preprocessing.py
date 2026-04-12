from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageChops, UnidentifiedImageError


MAX_DIMENSION = 1024
SIZE_DIVISOR = 28
SOURCE_IMAGE_DIR = "Image"
PROCESSED_IMAGE_DIR = "Image_processed"
SOURCE_CSV = "Train.csv"
PROCESSED_CSV = "Train_processed.csv"


def is_grayscale(image: Image.Image) -> bool:
	"""Return True if the image has no color information."""
	if image.mode in {"1", "L", "I", "I;16", "F", "LA"}:
		return True

	rgb = image.convert("RGB")
	r, g, b = rgb.split()
	return (
		ImageChops.difference(r, g).getbbox() is None
		and ImageChops.difference(r, b).getbbox() is None
	)


def compute_target_size(width: int, height: int) -> tuple[int, int]:
	"""Scale to max dimension and force both dimensions to be divisible by SIZE_DIVISOR."""
	longest_side = max(width, height)
	scale = min(1.0, MAX_DIMENSION / longest_side)

	scaled_w = max(1, int(width * scale))
	scaled_h = max(1, int(height * scale))

	new_w = (scaled_w // SIZE_DIVISOR) * SIZE_DIVISOR
	new_h = (scaled_h // SIZE_DIVISOR) * SIZE_DIVISOR

	# For very small sides, keep at least one divisible block.
	new_w = max(SIZE_DIVISOR, new_w)
	new_h = max(SIZE_DIVISOR, new_h)

	return new_w, new_h


def preprocess_dataset(train_dir: Path) -> None:
	image_dir = train_dir / SOURCE_IMAGE_DIR
	csv_path = train_dir / SOURCE_CSV
	processed_image_dir = train_dir / PROCESSED_IMAGE_DIR
	processed_csv_path = train_dir / PROCESSED_CSV

	if not image_dir.is_dir():
		raise FileNotFoundError(f"Image directory not found: {image_dir}")
	if not csv_path.is_file():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	processed_image_dir.mkdir(parents=True, exist_ok=True)

	with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		fieldnames = reader.fieldnames
		rows = list(reader)

	if not fieldnames:
		raise ValueError("Train.csv has no header")
	if "Image_name" not in fieldnames:
		raise ValueError("Train.csv must contain 'Image_name' column")

	processed_rows: list[dict[str, str]] = []
	grayscale_skipped = 0
	resized_count = 0
	copied_count = 0
	missing_count = 0
	unreadable_count = 0

	for row in rows:
		image_name = row["Image_name"].strip()
		image_path = image_dir / image_name
		processed_image_path = processed_image_dir / image_name

		if not image_path.is_file():
			missing_count += 1
			continue

		try:
			with Image.open(image_path) as img:
				if is_grayscale(img):
					grayscale_skipped += 1
					continue

				current_w, current_h = img.size
				target_w, target_h = compute_target_size(current_w, current_h)

				if (current_w, current_h) != (target_w, target_h):
					resized = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
					processed_image_path.parent.mkdir(parents=True, exist_ok=True)
					resized.save(processed_image_path)
					resized_count += 1
				else:
					processed_image_path.parent.mkdir(parents=True, exist_ok=True)
					shutil.copy2(image_path, processed_image_path)
					copied_count += 1

				new_row = dict(row)
				new_row["Image_name"] = image_name
				processed_rows.append(new_row)
		except UnidentifiedImageError:
			unreadable_count += 1

	with processed_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(processed_rows)

	print(f"Total rows in CSV: {len(rows)}")
	print(f"Rows written to processed CSV: {len(processed_rows)}")
	print(f"Grayscale images skipped: {grayscale_skipped}")
	print(f"Images resized: {resized_count}")
	print(f"Images copied without resize: {copied_count}")
	print(f"Missing image files referenced in CSV: {missing_count}")
	print(f"Unreadable image files: {unreadable_count}")
	print(f"Processed images directory: {processed_image_dir}")
	print(f"Processed CSV path: {processed_csv_path}")


def main() -> None:
	project_root = Path(__file__).resolve().parent
	train_dir = project_root / "Train"
	preprocess_dataset(train_dir)


if __name__ == "__main__":
	main()
