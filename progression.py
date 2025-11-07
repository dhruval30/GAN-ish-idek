import imageio
import os

image_dir = "output/generated_images"   
output_gif = "progress3.gif"

files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith(".png")],
    key=lambda x: int(''.join(filter(str.isdigit, x)))  
)

images = [imageio.imread(os.path.join(image_dir, f)) for f in files]

imageio.mimsave(output_gif, images, duration=0.15)  # 0.15s per frame
print(f"GIF saved as {output_gif}")