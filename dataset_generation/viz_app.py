import streamlit as st
from PIL import Image
from io import BytesIO
import datasets
import pandas as pd
import re

st.set_page_config(layout="wide", page_title="Fashion Edit Dataset Viewer")

st.title("Fashion Edit Dataset Viewer")

@st.cache_data
def load_dataset():
    dataset = datasets.load_dataset("AI-ServicesBB/fashion-edit-dataset", split="val")
    return dataset

def decode_image(image_bytes):
    return Image.open(BytesIO(image_bytes))

def highlight_text(text, search_term):
    if not search_term:
        return text
    pattern = re.compile(f'({re.escape(search_term)})', re.IGNORECASE)
    return pattern.sub(r'<span style="background-color: #FFFF00">\1</span>', text)

def clean_caption(caption):
    return caption.replace('Neutral Gray Background, ', '')

dataset = load_dataset()

dataset_list = [sample for sample in dataset]

st.sidebar.header("Filters")

all_statuses = sorted(list(set(sample.get('status', 'N/A') for sample in dataset_list)))
all_classes = sorted(list(set(sample.get('class_name', 'N/A') for sample in dataset_list)))

selected_status = st.sidebar.multiselect("Status", all_statuses, default=all_statuses)
selected_classes = st.sidebar.multiselect("Class", all_classes, default=[])

st.sidebar.markdown("### Text Search")
with st.sidebar.form("text_search"):
    edit_instruction_search = st.text_input("Search in Edit Instructions").lower()
    original_caption_search = st.text_input("Search in Original Captions").lower()
    resulting_caption_search = st.text_input("Search in Resulting Captions").lower()
    search_submitted = st.form_submit_button("Search")

if 'search_terms' not in st.session_state:
    st.session_state.search_terms = {
        'edit_instruction': '',
        'original_caption': '',
        'resulting_caption': ''
    }

if search_submitted:
    st.session_state.search_terms = {
        'edit_instruction': edit_instruction_search,
        'original_caption': original_caption_search,
        'resulting_caption': resulting_caption_search
    }

filtered_dataset = []
for sample in dataset_list:
    if sample.get('status', 'N/A') not in selected_status:
        continue
        
    if selected_classes and sample.get('class_name', 'N/A') not in selected_classes:
        continue
        
    if st.session_state.search_terms['edit_instruction'] and \
       st.session_state.search_terms['edit_instruction'] not in str(sample.get('edit_instruction', '')).lower():
        continue
    if st.session_state.search_terms['original_caption'] and \
       st.session_state.search_terms['original_caption'] not in str(sample.get('original_caption', '')).lower():
        continue
    if st.session_state.search_terms['resulting_caption'] and \
       st.session_state.search_terms['resulting_caption'] not in str(sample.get('resulting_caption', '')).lower():
        continue
        
    filtered_dataset.append(sample)

st.write(f"Dataset size: {len(filtered_dataset)} samples (filtered) out of {len(dataset)} total samples")

items_per_page = 20
total_pages = len(filtered_dataset) // items_per_page + (1 if len(filtered_dataset) % items_per_page > 0 else 0)

page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1) - 1
start_idx = page * items_per_page
end_idx = min(start_idx + items_per_page, len(filtered_dataset))

st.write(f"Showing items {start_idx + 1} to {end_idx} of {len(filtered_dataset)}")

for i in range(start_idx, end_idx):
    sample = filtered_dataset[i]
    
    st.markdown(f"### {i}. &nbsp;&nbsp;&nbsp; {sample.get('status', 'N/A')} | {sample.get('class_name', 'N/A')}")
    
    col1, col2, col3 = st.columns([4, 3, 4])
    
    with col1:
        st.write("Input Image")
        input_image = decode_image(sample["input_image"])
        st.image(input_image, use_container_width=True)
        highlighted_original = highlight_text(clean_caption(sample['original_caption']), 
                                           st.session_state.search_terms['original_caption'])
        st.markdown(highlighted_original, unsafe_allow_html=True)
        
    with col2:
        st.write("Edit Instruction")
        highlighted_instruction = highlight_text(sample['edit_instruction'], 
                                              st.session_state.search_terms['edit_instruction'])
        st.markdown(f"**{highlighted_instruction}**", unsafe_allow_html=True)
        
    with col3:
        st.write("Output Image")
        output_image = decode_image(sample["output_image"])
        st.image(output_image, use_container_width=True)
        highlighted_resulting = highlight_text(clean_caption(sample['resulting_caption']), 
                                            st.session_state.search_terms['resulting_caption'])
        st.markdown(highlighted_resulting, unsafe_allow_html=True)
    
    st.divider()
