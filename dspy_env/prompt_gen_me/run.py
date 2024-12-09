import json
import dspy
from loguru import logger


from prompt_gen_me.program import GenerateEditedCaptionModule

# _ = load_dotenv(find_dotenv())

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)

if __name__ == "__main__":
    original_caption = "A soft grey knit sweater with long, loose-fitting sleeves, a crew neckline, and ribbed cuffs and hem. The sweater drapes slightly over the hips, giving a relaxed silhouette."
    edit_instruction = "Crop the sweater at the waist and reshape the neckline into a deep V."
    module = GenerateEditedCaptionModule(original_caption, edit_instruction)
    edited_caption = module.forward()
    logger.info(f"Edited caption: {edited_caption}")
