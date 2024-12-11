import dspy
from .signature import Assess

class ValidateEditedCaptionModule(dspy.Module):
    """Module to validate edited captions."""

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(Assess)

    def forward(self, example, trace=None):
        is_edited_question = f"""
            On a scale from 1 to 10, how accurately does the assessed text reflect the described garment after the edit, 
            ensuring that all key features from the original caption: `{example.original_caption}` are correctly updated 
            based on the edit instruction: `{example.edit_instruction}`?
        """
        is_complete_question = f"""
            On a scale from 1 to 10, how well does the assessed text cover all relevant details from both the original caption: `{example.original_caption}` 
            and the edit instructions: `{example.edit_instruction}`, ensuring that no important garment features are missing?
        """
        is_same_words_question = f"""
            On a scale from 1 to 10, how closely does the assessed text match the original caption’s (`{example.original_caption}`) 
            structure, vocabulary, and phrasing, considering only necessary edits from the edit instructions: `{example.edit_instruction}`?
        """
        is_minimal_effort_question = f"""
            On a scale from 1 to 10, how well does the edit instructions: `{example.edit_instruction}` reflect a minimal-effort change,
            and is the scope of the assessed text appropriately limited to the described edit?
        """
    
        is_edited = self.assess(assessed_text=example.resulting_caption, assessment_question=is_edited_question)
        is_complete = self.assess(assessed_text=example.resulting_caption, assessment_question=is_complete_question)
        is_same_words = self.assess(assessed_text=example.resulting_caption, assessment_question=is_same_words_question)
        is_minimal_effort = self.assess(assessed_text=example.resulting_caption, assessment_question=is_minimal_effort_question)
        return is_edited, is_complete, is_same_words, is_minimal_effort
