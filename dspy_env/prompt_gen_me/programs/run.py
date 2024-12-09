import json
import dspy
from loguru import logger
import pandas as pd

from prompt_gen_me.programs.step1_generate_edit_caption_examples.program import GenerateEditCaptionExamplesModule
from prompt_gen_me.programs.step2_asses_generated_edit_caption_examples.program import ValidateEditedCaptionModule

# _ = load_dotenv(find_dotenv())

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
lm = dspy.LM('ollama_chat/qwen2.5:14b', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)

if __name__ == "__main__":
    task_directive = """
    Generate 8 new examples in the following JSON format. Each example should include:
	1.	original_caption: A detailed description of the upper-body garment’s visible features.
	2.	edit_instruction: A clear and concise description of minimal-effort modifications applied to the garment.
	3.	resulting_caption: A standalone, detailed description of the final edited garment reflecting only the visible changes.

    Ensure the output is in valid JSON format, and maintain consistency with the provided examples.
    No combination of original_caption and edit_instruction should be repeated.
    """
    module = GenerateEditCaptionExamplesModule(task_directive)
    generated_examples = module.forward()
    logger.info(generated_examples)

    results = []
    scores = []
    module = ValidateEditedCaptionModule()
    for example in json.loads(generated_examples):
        is_edited, is_consistent = module.forward(example)
        logger.info(f"\n\nIs edited: {is_edited.assessment_answer}\n{is_edited.reasoning}")
        logger.info(f"Is consistent: {is_consistent.assessment_answer}\n{is_consistent.reasoning}")

        edited, consistent = [m.assessment_answer for m in [is_edited, is_consistent]]
        score = edited + consistent
        scores.append(score)

        result_row = {
            'original_caption': example['original_caption'],
            'edit_instruction': example['edit_instruction'],
            'resulting_caption': example['resulting_caption'],
            'score': score,
            'is edited': is_edited.assessment_answer,
            'is edited: reason': is_edited.reasoning,
            'is consistent': is_consistent.assessment_answer, 
            'is consistent: reason': is_consistent.reasoning
        }
        results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

    print(f"\n{scores}\n")

