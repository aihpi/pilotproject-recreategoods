import streamlit as st
from PIL import Image, ImageFile
from streamlit_scroll_to_top import scroll_to_here
import os
import json
import uuid
from datetime import datetime
import zipfile
import io
import base64

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up history directory
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
HISTORY_FILE = os.path.join(HISTORY_DIR, "generation_history.json")

st.set_page_config(
    page_title="recreategoods",
    page_icon="🎨",
    layout="wide"
)

if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False

if st.session_state.scroll_to_top:
    scroll_to_here(0, key='top')
    st.session_state.scroll_to_top = False

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def save_generation(input_image, prompt, output_image):
    # Generate unique ID for this generation
    gen_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Get original filename without extension
    original_filename = st.session_state.last_uploaded_file
    base_filename = os.path.splitext(original_filename)[0]
    
    # Get input image extension
    input_ext = os.path.splitext(original_filename)[1].lstrip('.')
    if not input_ext:
        input_ext = "jpg"  # Default to jpg
    
    # Save input image
    input_filename = f"{base_filename}_{gen_id}_input.{input_ext}"
    input_path = os.path.join(HISTORY_DIR, input_filename)
    input_image.save(input_path)
    
    # Save output image
    output_filename = f"{base_filename}_{gen_id}_output.{input_ext}"
    output_path = os.path.join(HISTORY_DIR, output_filename)
    output_image.save(output_path)
    
    # Create history entry
    entry = {
        "id": gen_id,
        "timestamp": timestamp,
        "prompt": prompt,
        "model": selected_model,
        "input_image": input_filename,
        "output_image": output_filename,
        "original_filename": original_filename
    }
    
    # Load existing history
    history = load_history()
    history.append(entry)
    save_history(history)
    
    return entry

# Custom CSS for the title, steps, and history
st.markdown("""
    <style>
    .stMainBlockContainer {
        padding-top: 50px !important;
    }
    .title {
        font-size: 56px !important;
        font-weight: 700;
        letter-spacing: 2px;
        padding: 20px 0 50px !important;
    }
    .step-header {
        font-size: 20px;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    /* Hide file uploader name */
    .stFileUploaderFile, .st-emotion-cache-12xsiil {
        display: none !important;
    }
    .history-entry {
        margin-bottom: 20px;
    }
    .stButton p {
        font-size: 14px;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stMarkdown {
        height: 100%;
    }
    .stHorizontalBlock:has(.history) ~ .stHorizontalBlock > .stColumn:has(.stButton) {
        background: rgb(240, 242, 246);
        border-radius: 0.5rem;
        padding: 0 15px 15px;
        margin-bottom: 10px;
        max-width: 100%;
    }
    .history-prompt {
        font-size: 12px;
        color: #666;
        border-radius: 4px;
        height: 100%;
        align-items: center;
        line-height: 16px;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 15px;
    }
    .history-header {
        font-size: 14px;
        margin-bottom: 8px;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .refine-button {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 10px;
    }
    .refine-button p {
        margin: 0;
    }
    div.stButton button {
        min-width: 100px;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    /* Style for history item header buttons */
    div[data-testid="stHorizontalBlock"] div.stButton button {
        text-align: left;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        display: block;
        padding-right: 20px;
    }
    /* Keep action buttons centered */
    div[data-testid="column"]:has(div.stButton) button {
        text-align: center;
        min-width: 100px;
    }
    </style>
    <h1 class="title">r e c r e a t e g o o d s</h1>
""", unsafe_allow_html=True)

# Initialize session state for uploaded image if it doesn't exist
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Initialize session state for delete confirmation
if 'show_delete_confirmation' not in st.session_state:
    st.session_state.show_delete_confirmation = False

# Initialize session state for download confirmation
if 'show_download_confirmation' not in st.session_state:
    st.session_state.show_download_confirmation = False

def delete_all_history():
    # Clear the history file
    save_history([])
    # Delete all files in history directory except the history.json
    for filename in os.listdir(HISTORY_DIR):
        if filename != "generation_history.json":
            file_path = os.path.join(HISTORY_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.error(f"Error deleting {filename}: {e}")
    st.session_state.show_delete_confirmation = False
    st.rerun()

def create_download_zip():
    # Create a BytesIO object to store the zip file
    zip_buffer = io.BytesIO()
    
    # Create a new zip file
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add history.json
        zip_file.write(HISTORY_FILE, "history.json")
        
        # Add all image files
        for filename in os.listdir(HISTORY_DIR):
            if filename != "generation_history.json":
                file_path = os.path.join(HISTORY_DIR, filename)
                if os.path.isfile(file_path):
                    zip_file.write(file_path, filename)
    
    # Reset buffer position
    zip_buffer.seek(0)
    return zip_buffer

# Create three columns
col1, col2, col3 = st.columns([1, 1.2, 1])

# Column 1: Image Upload
with col1:
    st.markdown('<p class="step-header">Step 1: Upload Your Image of the Garment</p>', unsafe_allow_html=True)
    
    # Display image or placeholder
    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="Input Image", use_container_width=True)
    else:
        placeholder_height = 300
        st.markdown(
            f"""
            <div style="
                height: {placeholder_height}px;
                border: 2px dashed #cccccc;
                border-radius: 5px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #666666;
                padding: 20px;
            ">
                Input image will appear here
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Upload button below the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Handle file upload
    if uploaded_file is not None and (st.session_state.uploaded_image is None or uploaded_file.name != getattr(st.session_state, 'last_uploaded_file', None)):
        try:
            # Save the file name to track changes
            st.session_state.last_uploaded_file = uploaded_file.name
            # Convert to PIL Image
            image = Image.open(uploaded_file)
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            # Create a copy of the image to ensure it's loaded
            image_copy = image.copy()
            st.session_state.uploaded_image = image_copy
            st.rerun()
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

# Column 2: Edit Instructions and Model Selection
with col2:
    # Step 2: Text prompt
    st.markdown('<p class="step-header">Step 2: Enter Your Edit Instructions</p>', unsafe_allow_html=True)
    # Initialize prompt in session state if not present
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    prompt = st.text_area("Describe how you want to modify the garment", 
                         value=st.session_state.prompt,  # Use the session state value
                         height=100)
    # Keep prompt synchronized
    st.session_state.prompt = prompt
    
    # Step 3: Model selection
    st.markdown('<p class="step-header">Step 3: Choose Your Model</p>', unsafe_allow_html=True)
    model_options = [
        "Stable Diffusion",
        "DALL-E",
        "Midjourney Style"
    ]
    # Initialize selected_model in session state if not present
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Stable Diffusion"
    selected_model = st.selectbox("Select the AI model to use", 
                                model_options,
                                index=model_options.index(st.session_state.selected_model))
    # Keep selected_model synchronized
    st.session_state.selected_model = selected_model
    
    # Step 4: Generate
    st.markdown('<p class="step-header">Step 4: Generate</p>', unsafe_allow_html=True)
    if st.button("Generate Image", use_container_width=True):
        if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
            st.error("Please upload an image first!")
        elif not prompt:
            st.error("Please enter edit instructions!")
        else:
            with st.spinner("Generating image..."):
                # TODO: Add actual model inference here
                # For now, we'll just display the input image as a placeholder
                output_image = st.session_state.uploaded_image
                st.session_state.generated_image = output_image
                
                # Save the generation to history
                save_generation(
                    input_image=st.session_state.uploaded_image,
                    prompt=prompt,
                    output_image=output_image
                )

# Column 3: Output
with col3:
    st.markdown('<p class="step-header">Output</p>', unsafe_allow_html=True)
    if "generated_image" in st.session_state:
        st.image(st.session_state.generated_image, caption="Generated Image", use_container_width=True)
        
        # Add Refine button with arrow icon
        refine_col = st.columns([2, 1, 2])[1]  # Center the button
        with refine_col:
            st.markdown(
                """
                <style>
                div[data-testid="stHorizontalBlock"] div[data-testid="column"]:has(div.refine-button) {
                    display: flex;
                    justify-content: center;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            if st.button("↩ Refine", key="refine_button"):
                st.session_state.uploaded_image = st.session_state.generated_image
                del st.session_state.generated_image
                st.session_state.scroll_to_top = True
                st.rerun()
    else:
        placeholder_height = 300
        st.markdown(
            f"""
            <div style="
                height: {placeholder_height}px;
                border: 2px dashed #cccccc;
                border-radius: 5px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #666666;
                padding: 20px;
            ">
                Generated image will appear here
            </div>
            """,
            unsafe_allow_html=True
        )

# History Section
st.markdown("---")

# Load history first to check length
history = load_history()
history.reverse()  # Most recent first

# Create a row for history header and action buttons
header_col, buttons_col = st.columns([0.6, 0.4])

with header_col:
    st.markdown('<p class="step-header history">History</p>', unsafe_allow_html=True)

with buttons_col:
    
    # Create two buttons in a horizontal layout
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download", key="download", disabled=len(history) == 0):
            st.session_state.show_download_confirmation = True
    with col2:
        if st.button("Delete", key="delete", disabled=len(history) == 0):
            st.session_state.show_delete_confirmation = True

# Show download confirmation dialog if needed
if st.session_state.show_download_confirmation:
    with st.container():
        st.markdown("""
            <style>
            .download-confirmation {
                background-color: #e8f5e9;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            </style>
            <div class="download-confirmation">
                <h3>📥 Download History</h3>
                <p>Download a zip file containing all history entries and their associated images.</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Create zip file and download button
            zip_buffer = create_download_zip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if st.download_button(
                label="Yes, Download",
                data=zip_buffer,
                file_name=f"history_{timestamp}.zip",
                mime="application/zip",
                key="confirm_download"
            ):
                st.session_state.show_download_confirmation = False
                st.rerun()
        with col2:
            if st.button("Cancel", key="cancel_download"):
                st.session_state.show_download_confirmation = False
                st.rerun()

# Show delete confirmation dialog if needed
if st.session_state.show_delete_confirmation:
    with st.container():
        st.markdown("""
            <style>
            .delete-confirmation {
                background-color: #ffebee;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            </style>
            <div class="delete-confirmation">
                <h3>⚠️ Delete All History?</h3>
                <p>This action cannot be undone. All history entries and their associated images will be permanently deleted.</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete All", key="confirm_delete"):
                delete_all_history()
        with col2:
            if st.button("Cancel", key="cancel_delete"):
                st.session_state.show_delete_confirmation = False
                st.rerun()

# Display history in rows of 3
for i in range(0, len(history), 3):
    row_entries = history[i:i+3]
    cols = st.columns(3)
    
    for col_idx, entry in enumerate(row_entries):
        with cols[col_idx]:
            # Format timestamp
            dt = datetime.fromisoformat(entry["timestamp"])
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Display count, timestamp, and model
            full_header = f"#{len(history)-i-col_idx} - {formatted_time} ({entry.get('model', 'Unknown Model')})"
            
            # Add a clickable button that looks like the header with tooltip
            st.markdown(f"""
                <style>
                div[data-testid="stHorizontalBlock"] div.stButton button {{
                    cursor: pointer;
                }}
                </style>
            """, unsafe_allow_html=True)
            
            if st.button(full_header, key=f"history_{entry['id']}", use_container_width=True, help=full_header):
                # Restore the full entry state
                st.session_state.uploaded_image = Image.open(os.path.join(HISTORY_DIR, entry["input_image"]))
                st.session_state.last_uploaded_file = entry["original_filename"]
                st.session_state["prompt"] = entry["prompt"]
                st.session_state["selected_model"] = entry.get("model", "Stable Diffusion")
                st.session_state.generated_image = Image.open(os.path.join(HISTORY_DIR, entry["output_image"]))
                st.session_state.scroll_to_top = True
                st.rerun()
            
            # Create sub-columns for input, prompt, and output
            img1_col, prompt_col, img2_col = st.columns(3)
            
            with img1_col:
                input_path = os.path.join(HISTORY_DIR, entry["input_image"])
                if os.path.exists(input_path):
                    input_img = Image.open(input_path)
                    st.image(input_img, use_container_width=True)
            
            with prompt_col:
                st.markdown(
                    f"""<div class="history-prompt" title="{entry['prompt']}">{entry['prompt']}</div>""",
                    unsafe_allow_html=True
                )
            
            with img2_col:
                output_path = os.path.join(HISTORY_DIR, entry["output_image"])
                if os.path.exists(output_path):
                    output_img = Image.open(output_path)
                    st.image(output_img, use_container_width=True)
