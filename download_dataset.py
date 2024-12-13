import os
from datasets import load_dataset

ds = load_dataset("tomytjandra/h-and-m-fashion-caption-12k")
# Define output directory
output_dir = "h_and_m_fashion_caption_12k"
os.makedirs(output_dir, exist_ok=True)

# Iterate through the dataset and save images and captions
for i, sample in enumerate(ds["train"]):
    # Save the image
    image = sample["image"]
    image_path = os.path.join(output_dir, f"image_{i}.jpg")
    image.save(image_path, format="JPEG")

    # Save the caption
    caption = sample["text"]
    caption_path = os.path.join(output_dir, f"image_{i}.txt")
    with open(caption_path, "w") as f:
        f.write(caption)

print(f"Saved {len(ds['train'])} images and captions to {output_dir}")
