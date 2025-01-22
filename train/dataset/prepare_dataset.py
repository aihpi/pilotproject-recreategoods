import json
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    seeds = []
    with tqdm(desc="Listing dataset image seeds") as progress_bar:
        for prompt_dir in dataset_dir.iterdir():
            if prompt_dir.is_dir():
                # Collect only valid image files with supported extensions
                valid_images = list(prompt_dir.glob("*_0.*"))
                valid_images = [
                    img for img in valid_images if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                ]
                
                # Only process directories that contain at least one valid image
                if len(valid_images) > 0:
                    prompt_seeds = [image_path.name.split("_")[0] for image_path in sorted(valid_images)]
                    seeds.append((prompt_dir.name, prompt_seeds))
                    progress_bar.update()
    seeds.sort()

    # Write the filtered seeds to a JSON file
    with open(dataset_dir.joinpath("seeds.json"), "w") as f:
        json.dump(seeds, f)


if __name__ == "__main__":
    main()