import dspy

class Step1BootstrapFewShot(dspy.Signature):
    """
    Given a training set,
    generate n sets of x few-shot examples using the following techniques:
    - labeled few-shot from the training set
    - shuffled few-shot by generating outputs for the training set using the
    model and shuffling the outputs
    """

    trainset = dspy.InputField(
        desc="training set of examples"
    )
    few_shot_prompts: list[str] = dspy.OutputField(
        desc="list of few-shot prompts, "
    )


class GenerateExampleResponse(dspy.Signature):
    """
    Given an example input,
    generate a response using the model
    """

    original_caption: str = dspy.InputField(
        desc="original_caption"
    )
    edit_instruction: str = dspy.InputField(
        desc="edit_instruction"
    )
    resulting_caption: str = dspy.OutputField(
        desc="the edited caption"
    )
