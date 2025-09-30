import os
import pandas as pd
import shutil
from PIL import Image
import glob
import argparse
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import logging

def create_yolo_structure(output_path):
    """Create YOLO directory structure."""
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

def create_class_mapping():
    """Create mapping from class names to class indices."""
    classes = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return class_to_idx, classes

def get_image_dimensions(image_path):
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logging.error(f"Error reading image {image_path}: {e}")
        return None, None

def convert_to_yolo_format(base_path, output_path):
    """Convert Roboflow CSV dataset to YOLO format."""
    create_yolo_structure(output_path)
    class_to_idx, classes = create_class_mapping()
    splits = ['train', 'valid', 'test']

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn()) as progress:
        for split in splits:
            task = progress.add_task(f"[cyan]Processing {split}...", start=False)
            csv_path = os.path.join(base_path, split, '_classes.csv')
            if not os.path.exists(csv_path):
                logging.warning(f"{csv_path} not found. Skipping {split}.")
                continue

            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            expected_cols = ['filename'] + classes
            if list(df.columns) != expected_cols:
                logging.error(f"{split} CSV header mismatch.\nExpected: {expected_cols}\nGot: {list(df.columns)}")
                continue

            image_files = []
            image_dir = os.path.join(base_path, split)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(image_dir, ext)))
                image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

            progress.start_task(task)
            for _, row in progress.track(df.iterrows(), total=len(df), description=f"[green]Converting {split}"):
                filename = row['filename']
                image_path = next((img for img in image_files if os.path.basename(img) == filename), None)
                if not image_path:
                    logging.warning(f"Image {filename} not found in {split}. Skipping.")
                    continue

                width, height = get_image_dimensions(image_path)
                if width is None or height is None:
                    continue

                active_classes = [cls for cls in classes if cls in row and row[cls] == 1]
                if len(active_classes) != 1:
                    logging.warning(f"{filename}: Expected one active class but found {len(active_classes)}. Skipping.")
                    continue

                class_name = active_classes[0]
                class_idx = class_to_idx[class_name]

                dest_img = os.path.join(output_path, split, 'images', filename)
                shutil.copy2(image_path, dest_img)

                label_file = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(output_path, split, 'labels', label_file)
                with open(label_path, 'w') as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

def write_classes_file(output_path, classes):
    classes_file = os.path.join(output_path, 'classes.txt')
    with open(classes_file, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    logging.info(f"Classes file written: {classes_file}")

def write_yaml_file(output_path, classes):
    yaml_path = os.path.join(output_path, 'data.yaml')
    yaml_content = f"""# Indian Sign Language Dataset
path: {os.path.abspath(output_path)}
train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    logging.info(f"Data YAML file written: {yaml_path}")

def validate_dataset(output_path):
    """Validate that images and labels match."""
    logging.info("Validating dataset...")
    splits = ['train', 'valid', 'test']
    for split in splits:
        images_dir = os.path.join(output_path, split, 'images')
        labels_dir = os.path.join(output_path, split, 'labels')
        images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        logging.info(f"{split}: Images = {images}, Labels = {labels}")
        if images != labels:
            logging.warning(f"{split}: Number of images and labels do not match!")

def main():
    parser = argparse.ArgumentParser(description="Convert ISL dataset to YOLO format")
    parser.add_argument("--src", required=True, help="Source dataset folder containing train/valid/test")
    parser.add_argument("--dst", required=True, help="Destination folder for YOLO dataset")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s", handlers=[RichHandler()])

    base_path = args.src
    output_path = args.dst

    logging.info(f"Source dataset: {base_path}")
    logging.info(f"Output directory: {output_path}")

    convert_to_yolo_format(base_path, output_path)
    class_to_idx, classes = create_class_mapping()
    write_classes_file(output_path, classes)
    write_yaml_file(output_path, classes)
    validate_dataset(output_path)
    logging.info("Dataset conversion completed successfully!")

if __name__ == "__main__":
    main()
