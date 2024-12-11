import dspy

class GenerateEditCaptionExamples(dspy.Signature):
    """
    <prompt_requirements>

    Examples:
    <few_shot_examples>

    Task Directive:
    ”<task_directive>”

    """
    prompt_requirements = dspy.InputField(
        desc="A detailed description of the requirements for the edited caption.",
        type=str
    )
    few_shot_examples = dspy.InputField(
        desc="A list of few shot examples of the edited caption.",
        type=list
    )
    task_directive = dspy.InputField(
        desc="instruction",
        type=str
    )
    generated_examples = dspy.OutputField(
        desc="generated examples",
        type=list
    )
