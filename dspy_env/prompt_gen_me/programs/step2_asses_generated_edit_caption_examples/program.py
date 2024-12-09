import dspy
from .signature import Assess

class ValidateEditedCaptionModule(dspy.Module):
    """Module to validate edited captions."""

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(Assess)

    def forward(self, example, trace=None):
        is_edited_question = f"Is the assessed text mirroring that all provided edit instructions: `{example['edit_instruction']} are applied to the original caption: `{example['original_caption']}`?"
        is_consistent_question = f"Are all details of the original caption: `{example['original_caption']}` which were noch subject of edit instructions `{example['edit_instruction']}` still present in the assessed text?"
        
        is_edited = self.assess(assessed_text=example['resulting_caption'], assessment_question=is_edited_question)
        is_consistent = self.assess(assessed_text=example['resulting_caption'], assessment_question=is_consistent_question)

        return is_edited, is_consistent