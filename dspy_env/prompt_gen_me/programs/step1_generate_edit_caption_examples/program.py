import json
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
        prompt_requirements: str,
        few_shot_examples: list,
        task_directive: str
    ):
        self.prompt_requirements = prompt_requirements  
        self.few_shot_examples = few_shot_examples
        self.task_directive = task_directive
        self.signature = GenerateEditCaptionExamples
        self.generate_edit_caption_examples = dspy.ChainOfThought(
            GenerateEditCaptionExamples
        )

    def forward(self):
        logger.info("Generating edit caption examples")
        result = self.generate_edit_caption_examples(
            prompt_requirements=self.prompt_requirements,
            few_shot_examples=self.few_shot_examples,
            task_directive=self.task_directive
        )
        examples = []
        for example in json.loads(result.generated_examples):
            examples.append(dspy.Example(
                original_caption=example['original_caption'],
                edit_instruction=example['edit_instruction'],
                resulting_caption=example['resulting_caption']
            ))
        return examples