import dspy
import pandas as pd

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
lm = dspy.LM('ollama_chat/qwen2.5:14b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

module = dspy.ChainOfThought("caption, edit_instruction -> edited_caption")

def generate_edited_caption(caption, edit_instruction):
    result = module(caption=caption, edit_instruction=edit_instruction)
    print(f"\n\n-------------\nOriginal caption: {caption}")
    print(f"\nInstruction: {edit_instruction}")
    print(f"\nEdited caption: {result.edited_caption}")
    return result

class Assess(dspy.Signature):
    """Assess the quality of an edited caption along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

def validate_edited_caption(example, pred, trace=None):
    
    edited = f"The text should be the edited version of the original text: `{example.caption}`, edited with this instruction: `{example.edit_instruction}`. Is the assessed text edited according to the instruction?"
    consistent = f"Is the result consistent with the original description: `{example.caption}`, excpet the edited part: `{example.edit_instruction}`? Are all other details wich are not releated to the edit instructions still present in the assessed text?"
    
    edited =  dspy.ChainOfThought(Assess)(assessed_text=pred.edited_caption, assessment_question=edited)
    consistent = dspy.ChainOfThought(Assess)(assessed_text=pred.edited_caption, assessment_question=consistent)

    print(f"\nIs edited: {edited.assessment_answer}\n{edited.reasoning}")
    print(f"\nIs consistent: {consistent.assessment_answer}\n{consistent.reasoning}")

    return edited, consistent

examples_df = pd.read_csv('examples.csv')

examples = []
for _, row in examples_df.iterrows():
    example = dspy.Example(
        caption=row['caption'],
        edit_instruction=row['edit_instruction'], 
        edited_caption=row['edited_caption']
    ).with_inputs("caption", "edit_instruction")
    examples.append(example)

results = []
scores = []
for x in examples:
    pred = generate_edited_caption(**x.inputs())
    edited_assessment, consistent_assessment = validate_edited_caption(x, pred)

    edited, consistent = [m.assessment_answer for m in [edited_assessment, consistent_assessment]]
    score = edited + consistent
    scores.append(score)
    
    # Append results to each example
    result_row = {
        'caption': x.caption,
        'edit_instruction': x.edit_instruction,
        'edited_caption': x.edited_caption,
        'llm edit': pred.edited_caption,
        'score': score,
        'is edited': edited_assessment.assessment_answer,
        'is edited: reason': edited_assessment.reasoning,
        'is consistent': consistent_assessment.assessment_answer, 
        'is consistent: reason': consistent_assessment.reasoning
    }
    results.append(result_row)

# Write results to a new CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)


print(f"\n{scores}\n")