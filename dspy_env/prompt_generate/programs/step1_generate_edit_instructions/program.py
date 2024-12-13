import dspy
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signature import GenerateEditInstructions

class GenerateEditInstructionsModule(dspy.Module):
    """
    Generate edit instructions for garments.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        original_caption: str
    ):
        self.original_caption = original_caption
        self.signature = GenerateEditInstructions
        self.generate_edit_instructions = dspy.ChainOfThought(
            GenerateEditInstructions
        )

    def forward(self):
        logger.info("Generating edit instructions")
        result = self.generate_edit_instructions(
            original_caption=self.original_caption
        )
        return result.edit_instructions
