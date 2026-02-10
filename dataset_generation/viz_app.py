import streamlit as st
import json
from pathlib import Path
from PIL import Image
from utils.config_loader import load_config
import argparse
import numpy as np
def parse_arguments():
    parser = argparse.ArgumentParser(description="Streamlit App for Dataset Viewer")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file")
    args, _ = parser.parse_known_args()  # Ignore unrecognized args for Streamlit
    return args

# Parse arguments
args = parse_arguments()
config = load_config(args.config_path)

# Set paths to data directories
DATA_DIR = config.output_dir

st.set_page_config(layout="wide")

def load_metadata_and_images(data_dir):
    """Traverse subdirectories to load metadata and associated images."""
    all_data = []
    for subdir in Path(data_dir).iterdir():
        if subdir.is_dir():
            metadata_file = subdir / "metadata.jsonl"
            prompt_file = subdir / "prompt.json"
            if metadata_file.exists() and prompt_file.exists():
                # Load prompt.json
                with open(prompt_file, "r") as pf:
                    prompt_data = json.load(pf).get("prompt", {})

                # Load metadata.jsonl
                with open(metadata_file, "r") as f:
                    for line in f:
                        try:
                            metadata = json.loads(line.strip())
                            seed = metadata.get("seed")
                            image_0 = subdir / f"{seed}_0.jpg"
                            image_1 = subdir / f"{seed}_1.jpg"

                            # Check if images exist
                            if image_0.exists() and image_1.exists():
                                metadata["image_0"] = image_0
                                metadata["image_1"] = image_1
                                metadata["prompt"] = prompt_data  # Include prompt.json data
                                all_data.append(metadata)
                            else:
                                print(f"Skipping due to missing files: {seed}")
                        except Exception as e:
                            print(f"Error loading metadata from {metadata_file}: {e}")
    print(f"Loaded {len(all_data)} entries from {data_dir}")
    return all_data



def filter_metadata(metadata):
    """Filter metadata to include only specified keys."""
    keys_to_display = [
        "seed", "clip_sim_0", "clip_sim_1", "clip_sim_dir", "clip_sim_image",
        "amplify_factor", "suppress_factor", "shared_factor", "p2p_threshold",
        "cfg_scale", "self_replace_steps"
    ]
    return {key: metadata[key] for key in keys_to_display if key in metadata}

@st.cache_data
def get_data():
    """Cache loaded data to optimize performance."""
    return load_metadata_and_images(DATA_DIR)

# Load data
data = get_data()

# Sidebar options
st.sidebar.header("Filter Options")
filter_keyword = st.sidebar.text_input("Filter by keyword in metadata", "")
st.sidebar.markdown(f"### Total Entries: {len(data)}")

def navigate(step):
    st.session_state.index = max(0, min(len(data) - 1, st.session_state.index + step))

# Filter data
if filter_keyword:
    data = [item for item in data if filter_keyword.lower() in json.dumps(item).lower()]
    st.sidebar.markdown(f"### Filtered Entries: {len(data)}")

if "stats" not in st.session_state:
    st.session_state.stats = {
        "clip_sim_dir": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "clip_sim_image": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "amplify_factor": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "suppress_factor": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "shared_factor": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "p2p_threshold": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "cfg_scale": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
        "self_replace_steps": {"min": float('inf'), "max": float('-inf'), "min_sum": 0.0, "min_count": 0, "max_sum": 0.0, "max_count": 0},
    }

def update_stats(metadata):
    """Update min, max, and running averages for each metadata key."""
    for key, values in st.session_state.stats.items():
        curr_min, curr_max = calculate_average(values)
        value = metadata[key]
        # Update min and max
        if value < values["min"]:
            values["min"] = value
        if value > values["max"]:
            values["max"] = value

        if value < curr_min:
             values["min_sum"] -= value
        else:
            values["min_sum"] += value
        values["min_count"] += 1
        if value > curr_max:
            values["max_sum"] += value
        else:
            values["max_sum"] -= value
        values["max_count"] += 1



def calculate_average(values):
    """Calculate average min and max."""
    avg_min = values["min_sum"] / values["min_count"] if values["min_count"] > 0 else 0.0
    avg_max = values["max_sum"] / values["max_count"] if values["max_count"] > 0 else 0.0
    return avg_min, avg_max

# Display data
st.title("Dataset Viewer: Images and Metadata")
st.markdown("View images side by side with their corresponding metadata.")
if not data:
    st.warning("No data to display. Adjust the filter or check your dataset.")
else:
    if "index" not in st.session_state:
        st.session_state.index = 0
    current_item = data[st.session_state.index]
    # Keyboard navigation logic
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowRight') {
                window.location.href = "/?index=" + (parseInt(new URL(window.location.href).searchParams.get("index") || 0) + 1);
            }
            if (event.key === 'ArrowLeft') {
                window.location.href = "/?index=" + Math.max(0, parseInt(new URL(window.location.href).searchParams.get("index") || 0) - 1);
            }
        });
        </script>
    """, unsafe_allow_html=True)
     # Sidebar navigation for index
    st.sidebar.write("Use Arrow Keys or Sidebar to Navigate")
    st.sidebar.button("⬅️ Previous", on_click=lambda: navigate(-1))
    st.sidebar.button("Next ➡️", on_click=lambda: navigate(1))
    st.sidebar.write("### Original Caption")
    st.sidebar.write(current_item["prompt"].get("original_caption", "N/A"))
    st.sidebar.write("### Edit Instruction")
    st.sidebar.write(current_item["prompt"].get("edit_instruction", "N/A"))
    st.sidebar.write("### Resulting Caption")
    st.sidebar.write(current_item["prompt"].get("resulting_caption", "N/A"))

    # Display current data
    st.markdown(f"### Entry {st.session_state.index + 1} of {len(data)}")
    # Display images in a row
    cols = st.columns(2)
    with cols[0]:
        st.image(Image.open(current_item["image_0"]), caption=f"original image", use_container_width=True, width=512)
    with cols[1]:
        st.image(Image.open(current_item["image_1"]), caption=f"resulting image", use_container_width=True, width=512)
    # Thumbs Up and Thumbs Down buttons
    st.write("### Rate the data")
    thumbs_cols = st.columns(2)
    with thumbs_cols[0]:
        if st.button("👍 Thumbs Up"):
            update_stats(current_item)
            st.success("Values updated with thumbs up!")

    with thumbs_cols[1]:
        if st.button("👎 Thumbs Down"):
            st.warning("No changes were made.")

    # Display current min and max values
    st.write("### Current Stats")
    for key, values in st.session_state.stats.items():
        avg_min, avg_max = calculate_average(values)
        st.write(f"**{key}**:")
        st.write(f"  - Found Min = {values['min']:.4f}, Avg Min = {avg_min:.4f}")
        st.write(f"  - Found Max = {values['max']:.4f}, Avg Max = {avg_max:.4f}")
    # Display metadata underneath the row of images
    st.markdown("#### Metadata")

    # Display the selected entry
    current_item = data[st.session_state.index]
    filtered_metadata = filter_metadata(current_item)
    st.json(filtered_metadata)

    
    

st.sidebar.info("Scroll down to view all entries.")