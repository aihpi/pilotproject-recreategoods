#!/usr/bin/env python
import os
import sys

# Set cache directory environment variables BEFORE importing any HF modules
cache_dir = "/workspace/hf_cache"
print(f"Setting Hugging Face cache to: {cache_dir}")

# Make sure the directory exists
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(os.path.join(cache_dir, "transformers"), exist_ok=True)
os.makedirs(os.path.join(cache_dir, "datasets"), exist_ok=True)

# Set all relevant environment variables
os.environ["HF_HOME"] = cache_dir
os.environ["HF_CACHE_HOME"] = cache_dir  # Might be used in some versions
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Now import and run the main module
print("Cache directory setup complete. Importing main module...")

# Run the command passed in arguments
if len(sys.argv) > 1:
    # Join all arguments into a command string
    cmd = " ".join(sys.argv[1:])
    print(f"Running command: {cmd}")
    exit_code = os.system(cmd)
    sys.exit(exit_code >> 8)  # Extract the exit code
else:
    print("No command provided. Set_cache.py expects a command to run.")
    sys.exit(1) 