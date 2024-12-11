import json
import dspy
from loguru import logger
import pandas as pd

from prompt_gen_me.programs.step0_generate_edited_caption.program import GenerateEditedCaptionModule
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
	3.	resulting_caption: A standalone, detailed description of the final edited garment reflecting the visible changes. Keep the original caption’s structure and vocabulary wherever possible, while reflecting the required edit.”.

    Ensure the output is in valid JSON format, and maintain consistency with the provided examples.
    No combination of original_caption and edit_instruction should be repeated.
    """
    with open("prompt_gen_me/data/examples_long_0.json", "r", encoding="utf-8") as f:
        few_shot_examples = [dspy.Example(**example) for example in json.loads(f.read())]

    if True:
        generated_examples = few_shot_examples
        for example in few_shot_examples:
            module = GenerateEditedCaptionModule(example.original_caption, example.edit_instruction)
            example.edited_caption = module.forward()
    else:
        generated_examples = module.forward()
        logger.info(generated_examples)

    results = []
    scores = []
    module = ValidateEditedCaptionModule()
    for example in generated_examples:
        is_edited, is_consistent, is_same_words, is_minimal_effort = module.forward(example)
        edited, consistent, same_words, minimal_effort = [m.assessment_answer for m in [is_edited, is_consistent, is_same_words, is_minimal_effort]]
        score = edited + consistent + same_words + minimal_effort
        scores.append(score)

        result_row = {
            'original_caption': example['original_caption'],
            'edit_instruction': example['edit_instruction'],
            'resulting_caption': example['resulting_caption'],
            'score': score,
            'is edited': is_edited.assessment_answer,
            'is edited: reason': is_edited.reasoning,
            'is consistent': is_consistent.assessment_answer, 
            'is consistent: reason': is_consistent.reasoning,
            'is same words': is_same_words.assessment_answer,
            'is same words: reason': is_same_words.reasoning,
            'is minimal effort': is_minimal_effort.assessment_answer,
            'is minimal effort: reason': is_minimal_effort.reasoning,
        }
        results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

    print(f"\n{scores}\n")

