import dspy
from loguru import logger
import pandas as pd
import argparse

from prompt_generate.programs.personas import personas

from prompt_generate.programs.step0_generate_original_captions.program import GenerateOriginalCaptionsModule
from prompt_generate.programs.step1_generate_edit_instructions.program import GenerateEditInstructionsModule
from prompt_generate.programs.step2_generate_edited_caption.program import GenerateEditedCaptionModule
from prompt_generate.programs.step3_asses_generated_edit_caption_examples.program import ValidateEditedCaptionModule

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

    unique_original_captions = set()
        
    for persona in personas:
        module = GenerateOriginalCaptionsModule(persona)
        original_captions = module.forward()
        num_before = len(unique_original_captions)
        unique_original_captions.update(original_captions)
        num_after = len(unique_original_captions)
        logger.info(f"Added {num_after - num_before} new captions")

    logger.info(f"Total unique original captions: {len(unique_original_captions)}")

    captions_and_instructions = []
    for caption in unique_original_captions:
        print(f"\n{caption}")
        module = GenerateEditInstructionsModule(caption)
        edit_instructions = module.forward()
        for instruction in edit_instructions:
            print(instruction)
            captions_and_instructions.append({
                'original_caption': caption,
                'edit_instruction': instruction
            })

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

    for input in captions_and_instructions:
        module = GenerateEditedCaptionModule(input['original_caption'], input['edit_instruction'], prompt_requirements, few_shot_examples)
        input['resulting_caption'] = module.forward()

    results = []
    scores = []
    sum_score = 0
    max_score = 0
    count = 0
    module = ValidateEditedCaptionModule()
    for example in captions_and_instructions:
        is_edited, is_complete, is_same_words, is_minimal_effort = module.forward(example)
        edited, complete, same_words, minimal_effort = [m.assessment_answer for m in [is_edited, is_complete, is_same_words, is_minimal_effort]]
        score = edited + complete + same_words + minimal_effort
        count += 1
        print(f"\n{count}.\n{example['original_caption']}\n{example['resulting_caption']}\n{score} / {edited} + {complete} + {same_words} + {minimal_effort}\n")
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

