from huggingface_hub import HfApi
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--path_in_repo", type=str, required=True)
    
    args = parser.parse_args()

    api = HfApi()
    api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
    api.upload_folder(
    folder_path=args.folder_path,
    repo_id=args.repo_id,
    path_in_repo=args.path_in_repo,
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
    )

if __name__ == "__main__":
    main()