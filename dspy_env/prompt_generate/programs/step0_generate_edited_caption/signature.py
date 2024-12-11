import dspy

class GenerateEditedCaption(dspy.Signature):
    """
    <prompt_requirements>

    Examples:
    <few_shot_examples>
    
    Ensure the output is maintain consistency with the provided examples.
    """

    original_caption: str = dspy.InputField(
        desc="A detailed description of the upper-body garment’s visible features."
    )
    edit_instruction: str = dspy.InputField(
        desc="A clear and concise description of minimal-effort modifications applied to the garment."
    )
    prompt_requirements: str = dspy.InputField(
        desc="A detailed description of the requirements for the resulting caption."
    )
    few_shot_examples: list = dspy.InputField(
        desc="A list of few shot examples."
    )
    resulting_caption: str = dspy.OutputField(
        desc="A standalone, detailed description of the final edited garment reflecting visible changes. Ensure clarity, completeness, and consistency."
    )
