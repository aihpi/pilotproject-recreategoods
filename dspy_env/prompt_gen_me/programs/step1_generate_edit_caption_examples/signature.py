import dspy

class GenerateEditCaptionExamples(dspy.Signature):
    """
    Persona: 
    Lila Verona, a sustainable fashion designer, reworks leftover garments into stylish upper-body pieces with minimal effort. 
    Her captions precisely describe visible garment details, and the resulting captions reflect only the final appearance of the edited garment, suitable for image generation.
    
    Examples:
    <few_shot_examples>

    Task Directive:
    ”<task_directive>”

    """
    task_directive = dspy.InputField(
        desc="instruction",
        type=str
    )
    generated_examples = dspy.OutputField(
        desc="generated examples",
        type=list
    )
