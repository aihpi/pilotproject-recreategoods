from pathlib import Path
from PIL import Image
from io import BytesIO
import datasets

# Load the dataset
dataset = datasets.load_dataset("AI-ServicesBB/fashion-edit-dataset", split="val")

# Function to decode raw bytes to a PIL image
def decode_image(image_bytes):
    """
    Decode raw bytes into a PIL Image.

    Args:
        image_bytes (bytes): The raw bytes of an image.

    Returns:
        PIL.Image.Image: The decoded PIL Image.
    """
    return Image.open(BytesIO(image_bytes))

# Example: Load and display the first sample
sample = dataset[0]  # Access the first sample

# Decode input and output images
input_image = decode_image(sample["input_image"])
output_image = decode_image(sample["output_image"])

# Print metadata and display images
print("Original Caption:", sample["original_caption"])
print("Edit Instruction:", sample["edit_instruction"])
print("Resulting Caption:", sample["resulting_caption"])
print("Status:", sample["status"])

# Display the images
# input_image.show(title="Input Image")
# output_image.show(title="Output Image")
print(input_image)