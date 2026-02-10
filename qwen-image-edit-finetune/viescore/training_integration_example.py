"""
Example of how to integrate VIEScore evaluation into training pipeline

This shows how to use the VIEScoreTrainingHook with your training code.
"""

import sys
from pathlib import Path

# Add the viescore module to path
viescore_path = Path(__file__).parent
if str(viescore_path) not in sys.path:
    sys.path.insert(0, str(viescore_path))

from script import setup_viescore_training_hook, run_local_model_evaluation_pipeline


def example_training_loop_with_viescore():
    """
    Example training loop showing VIEScore integration
    """

    # Setup VIEScore hook
    print("Setting up VIEScore evaluation hook...")
    viescore_hook = setup_viescore_training_hook(
        validation_images_dir="../extracted_images",
        validation_instructions_file="../edit_instructions.csv",
        eval_frequency=1000,  # Evaluate every 1000 steps
        max_eval_samples=5,   # Use 5 validation samples per evaluation
    )

    # Simulated training loop
    for step in range(5000):
        # Your training code here
        # ...

        # Call VIEScore hook at the end of each step
        if step % 100 == 0:  # Print progress every 100 steps
            print(f"Training step {step}")

        # The hook will automatically evaluate when step % eval_frequency == 0
        viescore_hook.on_step_end(
            model_path=f"./models/checkpoint_step_{step}",
            step=step
        )

        # Get latest scores if available
        latest_scores = viescore_hook.get_latest_scores()
        if latest_scores:
            print(f"Latest VIEScore: {latest_scores['avg_overall']:.2f}/10")


def evaluate_trained_model(model_path: str, sample_size: int = 10):
    """
    Evaluate a specific trained model checkpoint

    Args:
        model_path: Path to the model checkpoint
        sample_size: Number of samples to evaluate
    """
    print(f"Evaluating model: {model_path}")

    results = run_local_model_evaluation_pipeline(
        model_path=model_path,
        sample_size=sample_size,
        use_viescore=True,
        output_dir=f"evaluation_results_{Path(model_path).name}"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VIEScore Training Integration Example")
    parser.add_argument("--mode", choices=["training", "evaluate"], default="training",
                       help="Mode: training simulation or model evaluation")
    parser.add_argument("--model_path", type=str, default="",
                       help="Path to model checkpoint for evaluation")
    parser.add_argument("--sample_size", type=int, default=10,
                       help="Number of samples for evaluation")

    args = parser.parse_args()

    if args.mode == "training":
        print("Running training loop simulation with VIEScore...")
        example_training_loop_with_viescore()

    elif args.mode == "evaluate":
        if not args.model_path:
            print("Please provide --model_path for evaluation mode")
            sys.exit(1)

        print("Running model evaluation...")
        evaluate_trained_model(args.model_path, args.sample_size)