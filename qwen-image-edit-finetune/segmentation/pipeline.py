#!/usr/bin/env python3
"""
Image Segmentation Pipeline using Fashionpedia Detector API
"""

import os
import json
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class FashionpediaSegmentationPipeline:
    def __init__(self, api_key: str, metadata_csv_path: str = None):
        self.api_key = api_key
        self.api_url = "https://chat.hpi-sci.de/api/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.metadata_df = None
        self.similarity_model = None

        # Load metadata CSV if provided
        if metadata_csv_path and Path(metadata_csv_path).exists():
            self.load_metadata(metadata_csv_path)

        # Initialize sentence transformer for similarity computation
        try:
            self.similarity_model = SentenceTransformer('google/embeddinggemma-300m')
            print("Loaded EmbeddingGemma model for similarity computation")
        except Exception as e:
            print(f"Warning: Could not load EmbeddingGemma model: {e}")
            print("Falling back to all-MiniLM-L6-v2...")
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded fallback sentence transformer model")
            except Exception as e2:
                print(f"Warning: Could not load any similarity model: {e2}")
                self.similarity_model = None

    def load_metadata(self, csv_path: str):
        """Load metadata CSV file"""
        try:
            self.metadata_df = pd.read_csv(csv_path)
            print(f"Loaded metadata with {len(self.metadata_df)} entries")
        except Exception as e:
            print(f"Error loading metadata CSV: {e}")
            self.metadata_df = None

    def get_prompt_for_image(self, image_name: str) -> Optional[str]:
        """Get the prompt for a given image from metadata"""
        if self.metadata_df is None:
            return None

        # Try different image path formats
        possible_paths = [
            f"control_images/{image_name}",
            f"images/{image_name}",
            image_name
        ]

        for path in possible_paths:
            match = self.metadata_df[self.metadata_df['edit_image'] == path]
            if not match.empty:
                return match.iloc[0]['prompt']

        return None

    def compute_similarity(self, categories: List[str], prompt: str) -> Dict[str, float]:
        """Compute cosine similarity between detected categories and prompt"""
        if not self.similarity_model or not categories or not prompt:
            return {}

        try:
            # Create comprehensive category descriptions for all 27 Fashionpedia categories
            category_descriptions = {
                # Main garments and structural elements
                "shirt": "shirt upper garment with body, collar, sleeves, and potential waist area for belts",
                "top": "top upper garment covering torso with potential for sleeves and waist modifications",
                "sweater": "sweater knitted upper garment with body, sleeves, and neckline area",
                "cardigan": "cardigan open-front sweater with buttons or closures and sleeves",
                "jacket": "jacket outer garment with body, sleeves, waist area for belts, and collar",
                "vest": "vest sleeveless garment worn over shirt with front opening and waist area",
                "coat": "coat outer garment with sleeves, collar, and body covering torso",
                "dress": "dress full-length garment covering torso and legs with waist area for belts",
                "jumpsuit": "jumpsuit one-piece garment combining top and bottom with waist area",
                "skirt": "skirt lower garment covering hips and legs with waist area for belts",
                "pants": "pants lower garment covering legs with waist area for belts and pockets",
                "shorts": "shorts short pants covering upper legs with waist area",
                "cape": "cape sleeveless outer garment draped over shoulders",

                # Garment parts and details
                "sleeve": "sleeve arm covering part of garment extending from shoulder to wrist",
                "collar": "collar neckline area and neck covering of garment",
                "lapel": "lapel folded collar area of jacket, coat, or formal garment",
                "neckline": "neckline neck opening and collar area of garment",
                "hood": "hood head covering attached to garment for protection",
                "pocket": "pocket functional compartment on clothing for storage and utility",
                "waistband": "waistband fabric band around waist area of pants or skirts",
                "hem": "hem finished bottom edge of garment preventing fraying",
                "cuff": "cuff finished end part of sleeve at wrist area",
                "belt": "belt waist accessory for cinching, fastening, and style",
                "scarf": "scarf neck accessory for warmth and decoration",
                "glove": "glove hand covering accessory for protection and warmth",
                "bag": "bag accessory for carrying items and storage",
                "shoe": "shoe foot covering and protection accessory"
            }

            # Use detailed descriptions or fallback to basic format
            category_texts = []
            for cat in categories:
                if cat in category_descriptions:
                    category_texts.append(category_descriptions[cat])
                else:
                    category_texts.append(f"fashion {cat} clothing item part")

            # Try to use EmbeddingGemma's specialized methods, fallback to standard encoding
            similarities = {}

            try:
                if hasattr(self.similarity_model, 'encode_query') and hasattr(self.similarity_model, 'encode_document'):
                    # Use EmbeddingGemma's specialized methods
                    query_embedding = self.similarity_model.encode_query(prompt)
                    document_embeddings = self.similarity_model.encode_document(category_texts)

                    # Use the model's similarity method directly
                    similarity_scores = self.similarity_model.similarity(query_embedding, document_embeddings)

                    for i, category in enumerate(categories):
                        similarities[category] = float(similarity_scores[0][i])
                else:
                    raise AttributeError("Model doesn't have specialized encoding methods")

            except Exception:
                # Fallback to standard encoding for compatibility
                all_texts = category_texts + [prompt]
                embeddings = self.similarity_model.encode(all_texts)

                prompt_embedding = embeddings[-1].reshape(1, -1)

                for i, category in enumerate(categories):
                    category_embedding = embeddings[i].reshape(1, -1)
                    similarity = cosine_similarity(category_embedding, prompt_embedding)[0][0]
                    similarities[category] = float(similarity)

            return similarities

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return {}

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def segment_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Segment a single image using the Fashionpedia API"""
        try:
            # Encode image
            image_b64 = self.encode_image(image_path)

            # Prepare request payload
            payload = {
                "model": "fashionpedia-detector",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ]
            }

            # Make API request
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def save_results(self, response: Optional[Dict[str, Any]], image_name: str, output_dir: Path, image_path: Path):
        """Save visualization and predictions for a single image"""
        if not response or 'fashionpedia' not in response:
            print(f"No fashionpedia data found for {image_name}")
            return

        fashionpedia_data = response['fashionpedia']

        # Save visualization
        if 'visualization_base64' in fashionpedia_data:
            viz_path = output_dir / 'visualizations' / f"{image_name}_segmented.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)

            with open(viz_path, 'wb') as f:
                f.write(base64.b64decode(fashionpedia_data['visualization_base64']))
            print(f"Visualization saved: {viz_path}")

        # Save JSON predictions (filtered to remove image data)
        if 'json' in fashionpedia_data:
            json_path = output_dir / 'predictions' / f"{image_name}_predictions.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)

            # Filter out the base64 image data to keep only predictions
            filtered_data = fashionpedia_data['json'].copy()
            if 'image' in filtered_data and isinstance(filtered_data['image'], dict):
                # Remove the large base64 image data but keep other image metadata
                if 'path' in filtered_data['image']:
                    filtered_data['image'] = {k: v for k, v in filtered_data['image'].items() if k != 'path'}
                    if not filtered_data['image']:  # If image dict becomes empty, remove it
                        del filtered_data['image']

            # Add similarity analysis if metadata is available
            similarity_data = self.add_similarity_analysis(filtered_data, image_path)
            if similarity_data:
                filtered_data['similarity_analysis'] = similarity_data

            with open(json_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Predictions saved: {json_path}")

    def add_similarity_analysis(self, predictions_data: Dict[str, Any], image_path: Path) -> Optional[Dict[str, Any]]:
        """Add similarity analysis between detected categories and prompt"""
        if self.metadata_df is None or not self.similarity_model:
            return None

        # Get prompt for this image
        image_name = image_path.name
        prompt = self.get_prompt_for_image(image_name)
        if not prompt:
            return None

        # Extract unique categories from predictions
        predictions = predictions_data.get('predictions', [])
        if not predictions:
            return None

        categories = list(set([pred.get('category_name', 'unknown') for pred in predictions]))
        categories = [cat for cat in categories if cat != 'unknown']

        if not categories:
            return None

        # Compute similarities
        similarities = self.compute_similarity(categories, prompt)

        # Create similarity analysis
        analysis = {
            'prompt': prompt,
            'detected_categories': categories,
            'category_similarities': similarities,
            'avg_similarity': np.mean(list(similarities.values())) if similarities else 0.0,
            'max_similarity': max(similarities.values()) if similarities else 0.0,
            'min_similarity': min(similarities.values()) if similarities else 0.0
        }

        print(f"Similarity analysis - Avg: {analysis['avg_similarity']:.3f}, Max: {analysis['max_similarity']:.3f}")
        return analysis

    def process_directory(self, input_dir: Path, output_dir: Path, limit: Optional[int] = None):
        """Process all images in a directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        if limit:
            image_files = image_files[:limit]

        print(f"Found {len(image_files)} images to process")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_path.name}")

            # Check if already processed - skip if prediction file exists
            json_path = output_dir / 'predictions' / f"{image_path.stem}_predictions.json"
            if json_path.exists():
                print(f"Skipping {image_path.name} - already processed")
                continue

            response = self.segment_image(image_path)
            if response:
                self.save_results(response, image_path.stem, output_dir, image_path)

            print(f"Completed {i}/{len(image_files)}")


def main():
    parser = argparse.ArgumentParser(description='Fashion Image Segmentation Pipeline')
    parser.add_argument('--input-dir',
                       default='/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/control_images/',
                       help='Input directory containing images')
    parser.add_argument('--output-dir',
                       default='./segmentation_output',
                       help='Output directory for results')
    parser.add_argument('--limit', type=int, default=10,
                       help='Limit number of images to process (default: 10)')
    parser.add_argument('--metadata-csv',
                       default='/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/metadata_edit.csv',
                       help='Path to metadata CSV file containing prompts')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('API_KEY')

    if not api_key or api_key == 'sk-your-api-key-here':
        print("Error: Please set your API_KEY in the .env file")
        return

    # Initialize pipeline
    pipeline = FashionpediaSegmentationPipeline(api_key, args.metadata_csv)

    # Process images
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    pipeline.process_directory(input_dir, output_dir, args.limit)
    print("Pipeline completed!")


if __name__ == "__main__":
    main()