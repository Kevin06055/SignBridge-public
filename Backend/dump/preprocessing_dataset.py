from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import shutil
import os

def apply_otsu_thresholding(image_array):
    """
    Apply Otsu's thresholding algorithm to find optimal threshold
    """
    hist, bin_edges = np.histogram(image_array.flatten(), bins=256, range=(0, 255))
    total_pixels = image_array.size
    current_max = 0
    threshold = 0
    
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0
    weight_background = 0
    
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    
    return threshold

def preprocess_image(image_path):
    """
    Preprocess image with Gaussian blur and Otsu thresholding
    """
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Apply Gaussian blur to reduce noise
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Convert to numpy array
    img_array = np.array(img_blurred)
    
    # Apply Otsu thresholding
    threshold = apply_otsu_thresholding(img_array)
    
    # Create binary mask
    binary_mask = (img_array > threshold).astype(np.uint8) * 255
    
    # Convert back to PIL image and then to RGB
    binary_img = Image.fromarray(binary_mask).convert('RGB')
    
    return binary_img

def process_dataset(input_data_path, output_data_path):
    """
    Process YOLO dataset with structure: data/train/images, data/valid/images, etc.
    """
    input_data_path = Path(input_data_path)
    output_data_path = Path(output_data_path)
    
    # Define splits - using 'valid' instead of 'val' as per your structure
    splits = ['train', 'valid', 'test']
    
    total_processed = 0
    
    for split in splits:
        # Define paths for this split
        images_input = input_data_path / split / 'images'
        labels_input = input_data_path / split / 'labels'
        images_output = output_data_path / split / 'images'
        labels_output = output_data_path / split / 'labels'
        
        # Skip if images folder doesn't exist
        if not images_input.exists():
            print(f"⚠️  Skipping {split} - folder {images_input} does not exist")
            continue
        
        # Create output directories
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_input.glob(ext)))
            image_files.extend(list(images_input.glob(ext.upper())))  # Handle uppercase
        
        if not image_files:
            print(f"⚠️  No images found in {images_input}")
            continue
        
        print(f"📁 Processing {len(image_files)} images in {split} split...")
        
        processed_count = 0
        for i, image_file in enumerate(image_files):
            try:
                # Process image with binary masking and thresholding
                processed_img = preprocess_image(image_file)
                
                # Save processed image
                output_image_path = images_output / image_file.name
                processed_img.save(output_image_path, quality=95)
                
                # Copy corresponding label file
                label_file = labels_input / f"{image_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, labels_output / label_file.name)
                else:
                    print(f"⚠️  Label file not found: {label_file}")
                
                processed_count += 1
                
                # Progress indicator
                if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                    print(f"   ✓ Processed {i + 1}/{len(image_files)} images ({((i+1)/len(image_files)*100):.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error processing {image_file}: {str(e)}")
                continue
        
        print(f"✅ Completed {split} split: {processed_count}/{len(image_files)} images processed")
        total_processed += processed_count
    
    # Copy data.yaml if it exists
    data_yaml = input_data_path / 'data.yaml'
    if data_yaml.exists():
        # Update paths in data.yaml for the new dataset
        with open(data_yaml, 'r') as f:
            yaml_content = f.read()
        
        # Replace old paths with new paths
        yaml_content = yaml_content.replace('data/train', 'data-albu/train')
        yaml_content = yaml_content.replace('data/valid', 'data-albu/valid')
        yaml_content = yaml_content.replace('data/test', 'data-albu/test')
        
        with open(output_data_path / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        print("📄 Copied and updated data.yaml configuration file")
    else:
        print("⚠️  data.yaml not found - you may need to create one for training")
    
    return total_processed

def verify_dataset_structure(data_path):
    """
    Verify and display the current dataset structure
    """
    data_path = Path(data_path)
    print(f"\n🔍 Dataset Structure Analysis for: {data_path}")
    print("=" * 50)
    
    if not data_path.exists():
        print(f"❌ Dataset path {data_path} does not exist!")
        return False
    
    splits = ['train', 'valid', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        images_path = data_path / split / 'images'
        labels_path = data_path / split / 'labels'
        
        if images_path.exists():
            image_count = len(list(images_path.glob('*.[jJ][pP][gG]'))) + \
                         len(list(images_path.glob('*.[pP][nN][gG]'))) + \
                         len(list(images_path.glob('*.[jJ][pP][eE][gG]')))
            label_count = len(list(labels_path.glob('*.txt'))) if labels_path.exists() else 0
            
            print(f"📂 {split}:")
            print(f"   Images: {image_count} files in {images_path}")
            print(f"   Labels: {label_count} files in {labels_path}")
            
            total_images += image_count
            total_labels += label_count
        else:
            print(f"❌ {split}: Missing images folder at {images_path}")
    
    print(f"\n📊 Total: {total_images} images, {total_labels} labels")
    return total_images > 0

def main():
    """
    Main function to run the preprocessing
    """
    # Dataset paths
    input_data_path = 'data'
    output_data_path = 'data-albu'
    
    print("🚀 Indian Sign Language Dataset Preprocessor")
    print("=" * 50)
    print("📋 Features:")
    print("   • Binary masking with Otsu thresholding")
    print("   • Gaussian blur for noise reduction")
    print("   • Perfect lighting condition normalization")
    print("   • YOLO format preservation")
    
    # Verify dataset structure first
    if not verify_dataset_structure(input_data_path):
        print("\n❌ Please check your dataset structure and try again.")
        return False
    
    print(f"\n🔄 Starting preprocessing...")
    print(f"📥 Input:  {input_data_path}")
    print(f"📤 Output: {output_data_path}")
    
    try:
        total_processed = process_dataset(input_data_path, output_data_path)
        
        if total_processed > 0:
            print(f"\n🎉 SUCCESS! Preprocessed {total_processed} images")
            print(f"📁 New dataset saved in: {output_data_path}")
            print(f"\n🔥 Ready for fine-tuning!")
            print(f"   Command: yolo train model=best.pt data={output_data_path}/data.yaml")
        else:
            print(f"\n⚠️  No images were processed. Please check your dataset.")
            
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()
