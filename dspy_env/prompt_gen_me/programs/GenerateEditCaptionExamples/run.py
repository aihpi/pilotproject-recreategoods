import json
import dspy
from loguru import logger


from prompt_gen_me.programs.GenerateEditCaptionExamples.program import GenerateEditCaptionExamplesModule

# _ = load_dotenv(find_dotenv())

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)

if __name__ == "__main__":
    task_directive = """
    Generate 3 new examples in the following JSON format. Each example should include:
	1.	original_caption: A detailed description of the upper-body garment’s visible features.
	2.	edit_instruction: A clear and concise description of minimal-effort modifications applied to the garment.
	3.	resulting_caption: A standalone, detailed description of the final edited garment reflecting only the visible changes.

    Ensure the output is in valid JSON format, and maintain consistency with the provided examples.
    """
    module = GenerateEditCaptionExamplesModule(task_directive)
    generated_examples = module.forward()
    logger.info(f"Generated examples: {generated_examples}")

