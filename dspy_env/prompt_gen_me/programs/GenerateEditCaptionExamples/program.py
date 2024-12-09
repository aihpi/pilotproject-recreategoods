import dspy
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signature import GenerateEditCaptionExamples

class GenerateEditCaptionExamplesModule(dspy.Module):
    """
    Generate edit caption examples.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        task_directive: str
    ):
        self.task_directive = task_directive
        self.signature = GenerateEditCaptionExamples
        self.generate_edit_caption_examples = dspy.ChainOfThought(
            GenerateEditCaptionExamples
        )

    def forward(self):
        logger.info("Generating edit caption examples")
        result = self.generate_edit_caption_examples(
            task_directive=self.task_directive
        )
        generated_examples = result.generated_examples
        return generated_examples