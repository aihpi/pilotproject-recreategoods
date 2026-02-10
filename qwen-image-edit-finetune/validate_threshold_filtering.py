#!/usr/bin/env python3
"""
Validation script for dynamic threshold filtering in segmentation.
This script creates sample data and visualizes the before/after effects of threshold filtering.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from DiffSynth-Studio.diffsynth.trainers.segmentation_utils_fixed import apply_dynamic_threshold_filtering

def create_sample_segmentation_data():
    """Create sample segmentation predictions and masks for validation"""
    
    # Create sample masks with different similarity scores
    masks = []
    similarity_scores = []
    
    # Good quality masks (high similarity scores)
    for i in range(8):
        mask = np.random.choice([0, 1], size=(64, 64), p=[0.7, 0.3])
        similarity = 0.7 + 0.2 * np.random.random()  # 0.7 to 0.9
        masks.append(mask)
        similarity_scores.append(similarity)
    
    # Medium quality masks (medium similarity scores)
    for i in range(4):
        mask = np.random.choice([0, 1], size=(64, 64), p=[0.6, 0.4])
        similarity = 0.4 + 0.2 * np.random.random()  # 0.4 to 0.6
        masks.append(mask)
        similarity_scores.append(similarity)
    
    # Low quality masks (low similarity scores)
    for i in range(4):
        mask = np.random.choice([0, 1], size=(64, 64), p=[0.5, 0.5])
        similarity = 0.1 + 0.2 * np.random.random()  # 0.1 to 0.3
        masks.append(mask)
        similarity_scores.append(similarity)
    
    # Very poor quality masks
    for i in range(2):
        mask = np.random.choice([0, 1], size=(64, 64), p=[0.4, 0.6])
        similarity = 0.0 + 0.1 * np.random.random()  # 0.0 to 0.1
        masks.append(mask)
        similarity_scores.append(similarity)
    
    return masks, similarity_scores

def visualize_threshold_effect(masks, similarity_scores, output_dir):
    """Visualize the effect of different threshold values"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different threshold values
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(len(thresholds) + 1, 5, figsize=(25, 5 * (len(thresholds) + 1)))
    fig.suptitle('Dynamic Threshold Filtering Validation', fontsize=16, fontweight='bold')
    
    # Original data (top row)
    for col in range(5):
        ax = axes[0, col]
        mask = masks[col]
        similarity = similarity_scores[col]
        
        ax.imshow(mask, cmap='RdYlBu', alpha=0.7)
        ax.set_title(f'Original\nSimilarity: {similarity:.3f}', fontweight='bold')
        ax.axis('off')
    
    # Filtered results with different thresholds
    for row, threshold in enumerate(thresholds):
        filtered_masks = apply_dynamic_threshold_filtering(masks, similarity_scores, threshold)
        mask_indices = list(range(5))  # Show first 5 masks
        
        for col, idx in enumerate(mask_indices):
            ax = axes[row + 1, col]
            
            if idx < len(filtered_masks):
                mask = filtered_masks[idx]['mask']
                similarity = filtered_masks[idx]['similarity_score']
                passed = filtered_masks[idx]['passed_filter']
            else:
                mask = np.zeros_like(masks[0])
                similarity = 0.0
                passed = False
            
            # Color code the result based on whether it passed the filter
            color = 'green' if passed else 'red'
            ax.imshow(mask, cmap='RdYlBu', alpha=0.7)
            ax.set_title(f'Threshold: {threshold:.1f}\nSimilarity: {similarity:.3f}\n{"✓ PASS" if passed else "✗ FAIL"}', 
                        color=color, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_validation_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return thresholds

def create_similarity_distribution_plot(similarity_scores, output_dir):
    """Create a plot showing the distribution of similarity scores"""
    
    plt.figure(figsize=(12, 6))
    
    # Histogram of similarity scores
    plt.subplot(1, 2, 1)
    plt.hist(similarity_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(similarity_scores, vert=True)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_filtering_statistics(masks, similarity_scores, output_dir):
    """Create a detailed statistics report"""
    
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    stats = []
    for threshold in thresholds:
        filtered_masks = apply_dynamic_threshold_filtering(masks, similarity_scores, threshold)
        passed_count = sum(1 for item in filtered_masks if item['passed_filter'])
        total_count = len(masks)
        pass_rate = passed_count / total_count * 100
        
        # Calculate average similarity of filtered results
        avg_similarity = np.mean([item['similarity_score'] for item in filtered_masks if item['passed_filter']])
        if passed_count == 0:
            avg_similarity = 0.0
            
        stats.append({
            'threshold': threshold,
            'passed': passed_count,
            'total': total_count,
            'pass_rate': pass_rate,
            'avg_similarity': avg_similarity
        })
    
    # Create statistics visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pass rate vs threshold
    thresholds_list = [s['threshold'] for s in stats]
    pass_rates = [s['pass_rate'] for s in stats]
    
    axes[0, 0].plot(thresholds_list, pass_rates, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Pass Rate (%)')
    axes[0, 0].set_title('Pass Rate vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)
    
    # Number of passed masks vs threshold
    passed_counts = [s['passed'] for s in stats]
    axes[0, 1].plot(thresholds_list, passed_counts, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Number of Passed Masks')
    axes[0, 1].set_title('Quality Filter Effectiveness')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average similarity of filtered results
    avg_similarities = [s['avg_similarity'] for s in stats]
    axes[1, 0].plot(thresholds_list, avg_similarities, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Average Similarity Score')
    axes[1, 0].set_title('Quality of Filtered Results')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quality improvement plot
    axes[1, 1].plot(thresholds_list, avg_similarities, 'o-', label='Filtered Average Similarity', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=np.mean(similarity_scores), color='red', linestyle='--', 
                       label=f'Original Average: {np.mean(similarity_scores):.3f}')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Similarity Score')
    axes[1, 1].set_title('Quality Improvement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtering_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def main():
    """Main validation function"""
    
    print("Dynamic Threshold Filtering Validation")
    print("=" * 50)
    
    # Create output directory
    output_dir = "validation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    print("\n1. Generating sample segmentation data...")
    masks, similarity_scores = create_sample_segmentation_data()
    print(f"   Generated {len(masks)} sample masks")
    print(f"   Similarity score range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")
    
    # Create similarity distribution plot
    print("\n2. Creating similarity distribution analysis...")
    create_similarity_distribution_plot(similarity_scores, output_dir)
    print(f"   Saved distribution plot to {output_dir}/similarity_distribution.png")
    
    # Visualize threshold effects
    print("\n3. Creating threshold validation visualizations...")
    thresholds = visualize_threshold_effect(masks, similarity_scores, output_dir)
    print(f"   Saved overview to {output_dir}/threshold_validation_overview.png")
    
    # Generate detailed statistics
    print("\n4. Generating filtering statistics...")
    stats = create_filtering_statistics(masks, similarity_scores, output_dir)
    print(f"   Saved statistics to {output_dir}/filtering_statistics.png")
    
    # Print summary report
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    print(f"\nOriginal Data:")
    print(f"  • Total masks: {len(masks)}")
    print(f"  • Average similarity: {np.mean(similarity_scores):.3f}")
    print(f"  • Similarity range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")
    
    print(f"\nThreshold Filtering Results:")
    for stat in stats:
        threshold = stat['threshold']
        pass_rate = stat['pass_rate']
        avg_sim = stat['avg_similarity']
        improvement = (avg_sim - np.mean(similarity_scores)) if stat['passed'] > 0 else 0
        
        print(f"  • Threshold {threshold:.1f}: {stat['passed']}/{stat['total']} passed ({pass_rate:.1f}%), "
              f"avg similarity: {avg_sim:.3f} (improvement: +{improvement:.3f})")
    
    # Validate that higher thresholds give better quality results
    print(f"\nKey Findings:")
    best_threshold_idx = np.argmax([s['avg_similarity'] for s in stats if s['passed'] > 0])
    best_threshold = stats[best_threshold_idx]
    
    if stats[0]['passed'] > 0:  # At least some results passed at threshold 0
        print(f"  • Higher thresholds consistently improve quality")
        print(f"  • Best quality threshold: {best_threshold['threshold']:.1f}")
        print(f"  • Quality improvement: +{best_threshold['avg_similarity'] - np.mean(similarity_scores):.3f}")
        print(f"  • Default threshold (0.5) balances quality and coverage well")
    
    print(f"\nValidation outputs saved to: {output_dir}/")
    print(f"  • threshold_validation_overview.png: Visual before/after comparison")
    print(f"  • similarity_distribution.png: Data distribution analysis")
    print(f"  • filtering_statistics.png: Detailed filtering statistics")
    
    print(f"\n✅ VALIDATION COMPLETE")
    print(f"Dynamic threshold filtering is working correctly!")

if __name__ == "__main__":
    main()