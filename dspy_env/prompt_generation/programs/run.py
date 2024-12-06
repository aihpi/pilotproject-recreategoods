import json
import dspy
from loguru import logger
from dotenv import find_dotenv, load_dotenv
from prompt_generation.programs.step1_bootstrap_few_shot.program import (
    Step1BootstrapFewShotModule
)
_ = load_dotenv(find_dotenv())

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)


if __name__ == "__main__":

    logger.info("Loading examples dataset")
    with open("prompt_generation/examples.json", "r", encoding="utf-8") as f:
        trainset = [dspy.Example(**example) for example in json.loads(f.read())]
    logger.info(
        f"Loaded {len(trainset)} training examples"
    )

    # initialize the number of instructions to generate
    num_instructions = 2
    num_sets = 2

    # Run the bootstrap few-shot program to generate few-shot examples
    logger.info("Step 1: Running bootstrap few-shot program")
    bootstrap_few_shot_program = Step1BootstrapFewShotModule(
        trainset=trainset[:20],
        num_sets=num_sets,
        num_labeled_shots=5,
        num_shuffled_shots=3,
        metric="accuracy"
    )
    bootstrap_few_shot_examples = bootstrap_few_shot_program()
    logger.info(
        f"Generated {len(bootstrap_few_shot_examples)} few-shot examples"
    )

    logger.info("Bootstrap Few-Shot Examples:")
    for i, example_set in enumerate(bootstrap_few_shot_examples, 1):
        logger.info(f"Set {i}:")
        for j, example in enumerate(example_set, 1):
            logger.info(f"  Example {j}:")
            logger.info(f"    Original Caption: {example['original_caption']}")
            logger.info(f"    Edit Instruction: {example['edit_instruction']}")
            logger.info(f"    Resulting Caption: {example['resulting_caption']}")
        logger.info("---")