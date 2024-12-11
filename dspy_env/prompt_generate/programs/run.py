import json
import dspy
from loguru import logger
import pandas as pd
import argparse

from prompt_generate.programs.step0_generate_edited_caption.program import GenerateEditedCaptionModule
from prompt_generate.programs.step1_generate_edit_caption_examples.program import GenerateEditCaptionExamplesModule
from prompt_generate.programs.step2_asses_generated_edit_caption_examples.program import ValidateEditedCaptionModule

# _ = load_dotenv(find_dotenv())

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
lm = dspy.LM('ollama_chat/qwen2.5:14b', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)

parser = argparse.ArgumentParser()
parser.add_argument('--fewshot', default='examples_0.json', help='Path to few-shot examples file')
parser.add_argument('--input', help='Path to input examples file')
args = parser.parse_args()

fewshot = args.fewshot
input = args.input

data_dir = "prompt_generate/data"

if __name__ == "__main__":

    prompt_requirements = """
        Persona: 
        Lila Verona, a sustainable fashion designer, reworks leftover garments into stylish upper-body pieces with minimal effort. 
        Her captions precisely describe visible garment details, focusing only on the final appearance of the edited garment.

        Requirements:
        - Captions should be clear, visually descriptive, and reflect minimal-effort changes.
        - Ensure the resulting caption maintains structure, vocabulary, and phrasing consistency with the original.
        - Capture all relevant details without adding unrelated or imaginative features.

        Evaluation Criteria:
        - **Accuracy**: Are all garment features accurately updated based on the edit instruction?
        - **Completeness**: Are all relevant details from both the original and the edit included?
        - **Consistency**: Does the caption retain the same structure, vocabulary, and phrasing?
        - **Minimal Effort**: Are the described changes minimal in scope?
    """

    fewshot_file = f"{data_dir}/{fewshot}"
    logger.info(f"Loading few-shot examples from {fewshot_file}")
    with open(fewshot_file, "r", encoding="utf-8") as f:
        few_shot_examples = f.read()

    task_directive = """
        Generate 5 new examples in the following JSON format. Each example should include:
        1.	original_caption: A detailed description of the upper-body garment’s visible features.
        2.	edit_instruction: A clear and concise description of minimal-effort modifications applied to the garment.
        3.	resulting_caption: A standalone, detailed description of the final edited garment reflecting visible changes. Ensure clarity, completeness, and consistency.”.

        Ensure the output is in valid JSON format, and maintain consistency with the provided examples.
        No combination of original_caption and edit_instruction should be repeated.
    """

    if input:
        input_file = f"{data_dir}/{input}"
        logger.info(f"Loading input examples from {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            generated_examples = [dspy.Example(**example) for example in json.loads(f.read())]
        for example in generated_examples:
            module = GenerateEditedCaptionModule(example.original_caption, example.edit_instruction, prompt_requirements, few_shot_examples)
            example.edited_caption = module.forward()
    else:
        module = GenerateEditCaptionExamplesModule(prompt_requirements, few_shot_examples, task_directive)
        generated_examples = module.forward()

    results = []
    scores = []
    sum_score = 0
    max_score = 0
    module = ValidateEditedCaptionModule()
    for example in generated_examples:
        is_edited, is_complete, is_same_words, is_minimal_effort = module.forward(example)
        edited, complete, same_words, minimal_effort = [m.assessment_answer for m in [is_edited, is_complete, is_same_words, is_minimal_effort]]
        score = edited + complete + same_words + minimal_effort
        print(f"{score}\n{example.original_caption}\n{example.resulting_caption}\n")
        scores.append(score)
        sum_score += score
        max_score += 40
        result_row = {
            'original_caption': example['original_caption'],
            'edit_instruction': example['edit_instruction'],
            'resulting_caption': example['resulting_caption'],
            'score': score,
            'is edited': is_edited.assessment_answer,
            'is edited: reason': is_edited.reasoning,
            'is complete': is_complete.assessment_answer, 
            'is complete: reason': is_complete.reasoning,
            'is same words': is_same_words.assessment_answer,
            'is same words: reason': is_same_words.reasoning,
            'is minimal effort': is_minimal_effort.assessment_answer,
            'is minimal effort: reason': is_minimal_effort.reasoning,
        }
        results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

    print(f"\n{sum_score}/{max_score}\n")
    print(f"\n{scores}\n")

