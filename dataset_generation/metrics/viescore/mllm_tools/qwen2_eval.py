import os
import torch
from typing import List
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from transformers.image_utils import load_image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Qwen2VL:
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
        # Load the model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda"
        ).eval()

    @property
    def processor(self):
        return AutoProcessor.from_pretrained(self.model.name_or_path)
    
    def prepare_prompt(self, image_links: List[str] = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        # Create the conversation format for Qwen2VL
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * len(image_links) + [{"type": "text", "text": text_prompt}],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Load images (supporting both URLs and PIL Images)
        images =  [load_image(image_link) for image_link in image_links]
        inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def get_parsed_output(self, inputs):
        # Generate output using the model
        generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
        generated_text = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return generated_text


if __name__ == "__main__":
    model = Qwen2VL()
    # Provide a pair of images and a prompt
    prompt = model.prepare_prompt(
        [
            "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg",
            "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        ],
        "What are the differences between these two images?"
    )
    res = model.get_parsed_output(prompt)
    print("Result:\n", res)