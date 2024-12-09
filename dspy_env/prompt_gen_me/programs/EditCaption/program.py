import dspy
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signature import GenerateEditedCaption

class GenerateEditedCaptionModule(dspy.Module):
    """
    Generate an edited caption.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        original_caption: str,
        edit_instruction: str
    ):
        self.original_caption = original_caption
        self.edit_instruction = edit_instruction
        self.signature = GenerateEditedCaption
        self.generate_edited_caption = dspy.ChainOfThought(
            GenerateEditedCaption
        )

    def forward(self):
        logger.info("Generating edited caption")
        result = self.generate_edited_caption(
            original_caption=self.original_caption,
            edit_instruction=self.edit_instruction
        )
        resulting_caption = result.resulting_caption
        return resulting_caption