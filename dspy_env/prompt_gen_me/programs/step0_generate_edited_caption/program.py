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
        edit_instruction: str,
        prompt_requirements: str,
        few_shot_examples: list
    ):
        self.original_caption = original_caption
        self.edit_instruction = edit_instruction
        self.prompt_requirements = prompt_requirements
        self.few_shot_examples = few_shot_examples
        self.signature = GenerateEditedCaption
        self.generate_edited_caption = dspy.ChainOfThought(
            GenerateEditedCaption
        )

    def forward(self):
        logger.info("Generating edited caption")
        result = self.generate_edited_caption(
            original_caption=self.original_caption,
            edit_instruction=self.edit_instruction,
            prompt_requirements=self.prompt_requirements,
            few_shot_examples=self.few_shot_examples
        )

        return result.resulting_caption