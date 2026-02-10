#!/usr/bin/env python3
"""
Analyze similarity scores from batch segmentation results
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_similarity_data(predictions_dir: Path) -> List[Dict[str, Any]]:
    """Load similarity analysis data from all prediction files"""
    similarity_data = []

    json_files = list(predictions_dir.glob("*_predictions.json"))
    print(f"Found {len(json_files)} prediction files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'similarity_analysis' in data:
                analysis = data['similarity_analysis']
                analysis['image_name'] = json_file.stem.replace('_predictions', '')
                similarity_data.append(analysis)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Loaded similarity data from {len(similarity_data)} files")
    return similarity_data


def analyze_similarities(similarity_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze similarity scores and calculate statistics"""
    all_scores = []
    all_max_scores = []
    all_avg_scores = []
    category_scores = {}

    for data in similarity_data:
        # Collect all individual similarity scores
        if 'category_similarities' in data:
            scores = list(data['category_similarities'].values())
            all_scores.extend(scores)

            # Track scores by category
            for category, score in data['category_similarities'].items():
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)

        # Collect aggregate scores
        if 'max_similarity' in data:
            all_max_scores.append(data['max_similarity'])
        if 'avg_similarity' in data:
            all_avg_scores.append(data['avg_similarity'])

    # Calculate overall statistics
    all_scores = np.array(all_scores)
    all_max_scores = np.array(all_max_scores)
    all_avg_scores = np.array(all_avg_scores)

    # Calculate top 20% threshold and average
    top_20_threshold = np.percentile(all_scores, 80)
    top_20_scores = all_scores[all_scores >= top_20_threshold]
    top_20_avg = np.mean(top_20_scores)

    # Count images that would have mask guidance at different thresholds
    def count_images_with_guidance(threshold):
        count = 0
        for data in similarity_data:
            if 'category_similarities' in data:
                max_score = max(data['category_similarities'].values())
                if max_score >= threshold:
                    count += 1
        return count

    # Test different thresholds
    threshold_analysis = {}
    test_thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    for thresh in test_thresholds:
        images_with_guidance = count_images_with_guidance(thresh)
        threshold_analysis[thresh] = {
            'images_with_guidance': images_with_guidance,
            'percentage_of_images': (images_with_guidance / len(similarity_data)) * 100,
            'scores_above_threshold': int(np.sum(all_scores >= thresh)),
            'percentage_of_scores': (np.sum(all_scores >= thresh) / len(all_scores)) * 100
        }

    # Calculate category statistics
    category_stats = {}
    for category, scores in category_scores.items():
        scores_array = np.array(scores)
        category_stats[category] = {
            'count': len(scores),
            'mean': float(np.mean(scores_array)),
            'median': float(np.median(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array))
        }

    results = {
        'total_images': len(similarity_data),
        'total_similarity_scores': len(all_scores),
        'overall_statistics': {
            'all_scores_mean': float(np.mean(all_scores)),
            'all_scores_median': float(np.median(all_scores)),
            'all_scores_std': float(np.std(all_scores)),
            'all_scores_min': float(np.min(all_scores)),
            'all_scores_max': float(np.max(all_scores))
        },
        'top_20_percent': {
            'threshold': float(top_20_threshold),
            'count': len(top_20_scores),
            'average': float(top_20_avg),
            'percentage_of_total': (len(top_20_scores) / len(all_scores)) * 100
        },
        'max_scores_per_image': {
            'mean': float(np.mean(all_max_scores)),
            'median': float(np.median(all_max_scores)),
            'std': float(np.std(all_max_scores)),
            'min': float(np.min(all_max_scores)),
            'max': float(np.max(all_max_scores))
        },
        'avg_scores_per_image': {
            'mean': float(np.mean(all_avg_scores)),
            'median': float(np.median(all_avg_scores)),
            'std': float(np.std(all_avg_scores)),
            'min': float(np.min(all_avg_scores)),
            'max': float(np.max(all_avg_scores))
        },
        'category_statistics': category_stats,
        'threshold_analysis': threshold_analysis
    }

    return results


def print_summary(results: Dict[str, Any]):
    """Print a summary of the similarity analysis"""
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nDataset Size:")
    print(f"  • Total images analyzed: {results['total_images']}")
    print(f"  • Total similarity scores: {results['total_similarity_scores']}")

    print(f"\nOverall Similarity Scores:")
    overall = results['overall_statistics']
    print(f"  • Mean: {overall['all_scores_mean']:.3f}")
    print(f"  • Median: {overall['all_scores_median']:.3f}")
    print(f"  • Std Dev: {overall['all_scores_std']:.3f}")
    print(f"  • Range: {overall['all_scores_min']:.3f} - {overall['all_scores_max']:.3f}")

    print(f"\nTop 20% Similarity Scores:")
    top20 = results['top_20_percent']
    print(f"  • Threshold: {top20['threshold']:.3f}")
    print(f"  • Count: {top20['count']}")
    print(f"  • Average: {top20['average']:.3f}")
    print(f"  • Percentage: {top20['percentage_of_total']:.1f}%")

    print(f"\nMax Scores Per Image:")
    max_scores = results['max_scores_per_image']
    print(f"  • Mean: {max_scores['mean']:.3f}")
    print(f"  • Median: {max_scores['median']:.3f}")
    print(f"  • Range: {max_scores['min']:.3f} - {max_scores['max']:.3f}")

    print(f"\nTop Categories by Average Score:")
    category_stats = results['category_statistics']
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

    for i, (category, stats) in enumerate(sorted_categories[:10], 1):
        print(f"  {i:2d}. {category:15s}: {stats['mean']:.3f} (n={stats['count']:3d})")

    print(f"\nThreshold Analysis (Images with Mask Guidance):")
    threshold_analysis = results['threshold_analysis']
    for thresh in sorted(threshold_analysis.keys()):
        data = threshold_analysis[thresh]
        print(f"  • Threshold {thresh:.2f}: {data['images_with_guidance']:2d}/{len(results['category_statistics'])} images "
              f"({data['percentage_of_images']:4.1f}%) - {data['scores_above_threshold']} scores "
              f"({data['percentage_of_scores']:4.1f}%)")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze similarity scores from segmentation results')
    parser.add_argument('--predictions-dir',
                       default='./segmentation_output_parallel/predictions/',
                       help='Directory containing prediction JSON files')
    parser.add_argument('--output',
                       help='Output JSON file for detailed results')

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        print(f"Error: Predictions directory does not exist: {predictions_dir}")
        return

    # Load similarity data
    similarity_data = load_similarity_data(predictions_dir)
    if not similarity_data:
        print("No similarity data found!")
        return

    # Analyze similarities
    results = analyze_similarities(similarity_data)

    # Print summary
    print_summary(results)

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()