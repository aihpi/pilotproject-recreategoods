import json
import dspy
from loguru import logger
from dotenv import find_dotenv, load_dotenv
from prompt_generation.programs.step1_bootstrap_few_shot.program import (
    Step1BootstrapFewShotModule
)
from prompt_generation.programs.step2_bootstrap_instruction.program import (
    Step2GenerateInstructionModule
)
from prompt_generation.programs.step3_generate_final_prompt.program import (
    Step3GenerateFinalPromptModule
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

    # extract the program code of this program in this file
    logger.info("Extracting program code")
    with open(__file__, "r", encoding="utf-8") as file:
        program_code = file.read()
    logger.info(f"Extracted program code with {len(program_code)} characters")

    # Run the bootstrap instruction program to generate instructions
    logger.info("Step 2: Running bootstrap instruction program")
    bootstrap_instruction_program = Step2GenerateInstructionModule(
        few_shot_prompts=bootstrap_few_shot_examples,
        program_code=str(program_code),
        num_instructions=num_instructions
    )
    results = bootstrap_instruction_program()
    instructions = []
    for result in results:
        instructions.append(result["instruction"])
    logger.info(f"Generated {len(instructions)} instructions")
    logger.info("Instructions:")
    for i, instruction in enumerate(instructions, 1):
        logger.info(f"  Instruction {i}: {instruction}")

    # Run the generate final prompt program to generate a final prompt
    logger.info("Step 3: Running generate final prompt program")
    final_prompts = []
    for instruction, few_shot_examples in zip(
        instructions, bootstrap_few_shot_examples
    ):
        # convert few_shot_examples to a string
        few_shot_examples_str = ""
        for example in few_shot_examples:
            try:
                input_str = example["question"]
                output_str = example["answer"]
                few_shot_examples_str += (
                    f"Question: {input_str}\nExpected Answer: {output_str}\n\n"
                )
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.error(f"Error: {e}")
        generate_final_prompt_program = Step3GenerateFinalPromptModule(
            instruction=instruction,
            few_shot_examples=few_shot_examples_str
        )
        final_prompt = generate_final_prompt_program()
        final_prompts.append(final_prompt["final_prompt"])

    logger.info("Final prompts:")
    for i, prompt in enumerate(final_prompts, 1):
        logger.info(f"  Prompt {i}: {prompt}")