import dspy
import pandas as pd

lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

module = dspy.ChainOfThought("caption, edit_instruction -> edited_caption")

def generate_edited_caption(caption, edit_instruction):
    result = module(caption=caption, edit_instruction=edit_instruction)
    print(f"Original: {caption}")
    print(f"Instruction: {edit_instruction}")
    print(f"Edited: {result.edited_caption}")
    return result

# Define the signature for automatic assessments.
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

    print(edited)
    print(consistent)

    edited, consistent = [m.assessment_answer for m in [edited, consistent]]
    score = edited + consistent

    return score

examples_df = pd.read_csv('examples.csv')

examples = []
for _, row in examples_df.iterrows():
    example = dspy.Example(
        caption=row['caption'],
        edit_instruction=row['edit_instruction'], 
        edited_caption=row['edited_caption']
    ).with_inputs("caption", "edit_instruction")
    examples.append(example)

scores = []
for x in examples:
    pred = generate_edited_caption(**x.inputs())
    score = validate_edited_caption(x, pred)
    scores.append(score)

print(scores)