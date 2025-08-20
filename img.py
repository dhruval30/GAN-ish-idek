import cv2
import os
import re


# Folder where your images are stored
img_folder = "/Users/dhruval/Documents/GAN-ish-idek/output/generated_images"   # change this to your path
output_file = "/Users/dhruval/Documents/GAN-ish-idek/progression.mp4"

# Natural sort (epoch_1.png, epoch_2.png, ..., epoch_50.png)
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1

images = sorted(
    [img for img in os.listdir(img_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
    key=numerical_sort
)

# Read the first image to get dimensions
frame = cv2.imread(os.path.join(img_folder, images[0]))
height, width, layers = frame.shape

# Create a video writer (fps = frames per second)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_file, fourcc, 5, (width, height))  # 5 fps

# Add each image as a frame
for image in images:
    img_path = os.path.join(img_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print(f"Video saved as {output_file}")
