import dspy
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signature import GenerateOriginalCaptions

class GenerateOriginalCaptionsModule(dspy.Module):
    """
    Generate original captions for garments.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        persona: str
    ):
        self.persona = persona
        self.signature = GenerateOriginalCaptions
        self.generate_original_captions = dspy.ChainOfThought(
            GenerateOriginalCaptions
        )

    def forward(self):
        logger.info("Generating original captions")
        result = self.generate_original_captions(
            persona=self.persona
        )
        return result.original_captions
