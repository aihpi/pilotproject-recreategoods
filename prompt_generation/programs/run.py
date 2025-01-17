import dspy
from loguru import logger
import pandas as pd
import json

from programs.personas import personas

from programs.step0_generate_original_captions.program import GenerateOriginalCaptionsModule
from programs.step1_generate_edit_instructions.program import GenerateEditInstructionsModule
from programs.step2_generate_edited_caption.program import GenerateEditedCaptionModule
from programs.step3_asses_generated_edit_caption_examples.program import ValidateEditedCaptionModule

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
models = ['ollama_chat/qwen2.5:14b', 'ollama_chat/llama3.1:8b']
temperatures = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# default config
lm = dspy.LM(models[0], api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

data_dir = "prompt_generate/data"

if __name__ == "__main__":

    # Generate as many as possible unique original captions per persona (avg. 25)
    unique_original_captions_plus_metadata = {}
    for model in models:
        for temperature in temperatures:
            with dspy.context(lm=dspy.LM(model, temperature=temperature, api_base='http://localhost:11434', api_key='')):
                for persona in personas:
                    metadata = {
                        'model': model,
                        'temperature': temperature,
                        'persona': persona.split(',')[0].strip()
                    }
                    logger.info(f"Generating original captions for {metadata}.")
                    module = GenerateOriginalCaptionsModule(persona)
                    original_captions = module.forward()
                    num_before = len(unique_original_captions_plus_metadata)
                    new_captions = [caption for caption in original_captions if caption not in unique_original_captions_plus_metadata]
                    for caption in new_captions:
                        unique_original_captions_plus_metadata[caption] = metadata
                        print(caption)
                    num_after = len(unique_original_captions_plus_metadata)
                    logger.info(f"Added {num_after - num_before} new captions. Now {num_after}.")
    logger.info(f"Total unique original captions: {len(unique_original_captions_plus_metadata)}")
    
    # Generate for each unique original caption three different edit instructions
    captions_and_instructions_plus_metadata = []
    for caption in unique_original_captions_plus_metadata.keys():
        print(f"\n{caption}")
        module = GenerateEditInstructionsModule(caption)
        edit_instructions = module.forward()
        for instruction in edit_instructions:
            print(instruction)
            captions_and_instructions_plus_metadata.append({
                'original_caption': caption,
                'edit_instruction': instruction,
                'metadata': unique_original_captions_plus_metadata[caption]
            })

    # Generate the edited caption for each unique original caption and edit instruction
    for input in captions_and_instructions_plus_metadata:
        module = GenerateEditedCaptionModule(input['original_caption'], input['edit_instruction'])
        input['resulting_caption'] = module.forward()

    # Export captions and edit instructions to JSON
    output_data = []
    for index, item in enumerate(captions_and_instructions_plus_metadata):
        output_data.append({
            'index': index,
            'original_caption': item['original_caption'],
            'edit_instruction': item['edit_instruction'],
            'resulting_caption': item['resulting_caption'],
            'metadata': item['metadata']
        })
    
    with open('examples.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    # Assess the edited caption qu
    results = []
    scores = []
    sum_score = 0
    max_score = 0
    count = 0
    module = ValidateEditedCaptionModule()
    for example in captions_and_instructions_plus_metadata:
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
            'model': example['metadata']['model'],
            'temperature': example['metadata']['temperature'],
            'persona': example['metadata']['persona']
        }
        results.append(result_row)

    # Export all to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('examples.csv', index=False)

    print(f"\n{sum_score}/{max_score}\n")

