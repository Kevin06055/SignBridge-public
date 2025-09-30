import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Paths
split = "train"  # train/test/valid
data_dir = "dataset"  # replace with your dataset path
csv_file = os.path.join(data_dir, split, "_classes.csv")

# Load CSV
df = pd.read_csv(csv_file)

# Pick an example image
example = df.iloc[3]  # first row
filename = example['filename']
img_path = os.path.join(data_dir, split, filename)
img = Image.open(img_path)
draw = ImageDraw.Draw(img)

# Get image size
w, h = img.size

# Whole-image box (x_min, y_min, x_max, y_max)
bbox = [0, 0, w, h]
draw.rectangle(bbox, outline="red", width=4)

# Show image
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
plt.show()

# Print class index
class_idx = example[1:].tolist().index(1)
print("Class index for this image:", class_idx)
