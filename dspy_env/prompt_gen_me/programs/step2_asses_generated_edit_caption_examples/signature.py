import dspy

class Assess(dspy.Signature):
    """Assess the quality of an edited caption along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: int = dspy.OutputField()