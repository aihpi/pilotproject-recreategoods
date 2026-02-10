# OpenWebUI Evaluation Pipeline

## Overview

This pipeline evaluates Qwen-Image-Edit model performance using your OpenWebUI deployment at https://chat.hpi-sci.de with:
- **Qwen-Image-Edit** for image editing
- **Pixtral** for VIEScore evaluation

## How It Works

### Updated Architecture

Instead of using Hugging Face and OpenAI APIs, the pipeline now:

1. **Connects to OpenWebUI** using your API key from `.env`
2. **Sends image editing requests** to Qwen-Image-Edit via OpenWebUI's chat completions API
3. **Evaluates results** using Pixtral model (also via OpenWebUI) for VIEScore metrics
4. **Saves results** locally with scores and edited images

### Key Changes

| Component | Original | OpenWebUI Version |
|-----------|----------|-------------------|
| Image Editing | Hugging Face Diffusers | OpenWebUI API → Qwen-Image-Edit |
| VIEScore | OpenAI GPT-4o | OpenWebUI API → Pixtral |
| Authentication | OpenAI API Key | OpenWebUI API Key |
| Endpoint | Multiple APIs | Single OpenWebUI instance |

## Setup

### 1. Prerequisites

```bash
# Install required packages
pip install requests pandas pillow tqdm python-dotenv
```

### 2. API Key Configuration

Your API key is already in `.env`:
```
API_KEY=sk-6f9bb3e51e04475cbd848478e8bb8e4e
```

### 3. Test Connection

First, verify your connection and check available models:

```bash
python test_openwebui_connection.py
```

This will:
- Verify API key works
- List all available models
- Identify image editing and vision models
- Provide recommendations

## Usage

### Quick Test (3 samples)

```bash
python openwebui_evaluation_pipeline.py
```

### Full Evaluation

```python
from openwebui_evaluation_pipeline import run_openwebui_evaluation_pipeline

# Process all 126 images
results = run_openwebui_evaluation_pipeline(
    input_dir="extracted_images",
    instructions_file="edit_instructions.csv",
    output_dir="openwebui_evaluation_results",
    use_viescore=True,
    sample_size=None,  # Process all
    qwen_model="qwen-image-edit",  # Adjust based on test script output
    pixtral_model="pixtral"  # Adjust based on test script output
)
```

### Custom Configuration

```python
# If model names are different in your OpenWebUI
results = run_openwebui_evaluation_pipeline(
    sample_size=10,
    qwen_model="your-actual-qwen-model-name",
    pixtral_model="your-actual-pixtral-model-name"
)
```

## API Details

### Authentication
- Uses Bearer token authentication
- Header: `Authorization: Bearer YOUR_API_KEY`

### Image Editing Request
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Edit instruction"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
}]
```

### VIEScore Evaluation
- Sends both original and edited images to Pixtral
- Requests scores for:
  - Semantic Consistency (0-10)
  - Perceptual Quality (0-10)
  - Overall Score (0-10)

## Output Structure

```
openwebui_evaluation_results/
├── edited_images/          # Generated edited images
│   ├── 2_edited.png
│   ├── 3_edited.png
│   └── ...
└── evaluation_results.csv  # Scores and metadata
```

## Troubleshooting

### Connection Issues
1. Run `python test_openwebui_connection.py` first
2. Verify API key in `.env` is correct
3. Check you have access to https://chat.hpi-sci.de

### Model Not Found
- Run test script to see actual model names
- Update `qwen_model` and `pixtral_model` parameters accordingly

### Rate Limiting
- Add delays between requests if needed
- Process in smaller batches

### Image Format Issues
- OpenWebUI expects base64 encoded images
- Pipeline handles PNG/JPEG automatically

## Performance Notes

- Processing time depends on OpenWebUI cluster load
- Each evaluation makes 2 API calls (edit + score)
- Consider running in batches for large datasets
- Results are saved incrementally

## Example Output

After running, check results:

```python
import pandas as pd

# Load results
df = pd.read_csv("openwebui_evaluation_results/evaluation_results.csv")

# View average scores
print(f"Avg Semantic Consistency: {df['semantic_consistency'].mean():.2f}")
print(f"Avg Perceptual Quality: {df['perceptual_quality'].mean():.2f}")
print(f"Avg Overall Score: {df['overall'].mean():.2f}")

# Find best/worst edits
best = df.nlargest(5, 'overall')
worst = df.nsmallest(5, 'overall')
```