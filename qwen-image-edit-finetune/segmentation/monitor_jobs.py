#!/usr/bin/env python3
"""
Monitor parallel segmentation jobs
"""

import subprocess
import time
from pathlib import Path
import json


def count_processed_images(output_dir: str) -> int:
    """Count how many images have been processed"""
    predictions_dir = Path(output_dir) / 'predictions'
    if not predictions_dir.exists():
        return 0

    return len(list(predictions_dir.glob('*_predictions.json')))


def get_running_jobs() -> int:
    """Get number of running segmentation jobs"""
    try:
        result = subprocess.run(['squeue', '-u', subprocess.getoutput('whoami')],
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
        return len([line for line in lines if 'segment_' in line])
    except:
        return 0


def estimate_completion_time(processed: int, total: int, start_time: float) -> str:
    """Estimate completion time based on current progress"""
    if processed == 0:
        return "Calculating..."

    elapsed = time.time() - start_time
    rate = processed / elapsed  # images per second
    remaining = total - processed

    if rate > 0:
        eta_seconds = remaining / rate
        eta_hours = eta_seconds / 3600
        return f"{eta_hours:.1f} hours"
    else:
        return "Unknown"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor segmentation job progress')
    parser.add_argument('--output-dir', default='./segmentation_output_parallel',
                       help='Output directory to monitor')
    parser.add_argument('--total-images', type=int, default=200000,
                       help='Total number of images being processed')
    parser.add_argument('--update-interval', type=int, default=60,
                       help='Update interval in seconds')

    args = parser.parse_args()

    start_time = time.time()

    print(f"Monitoring segmentation jobs...")
    print(f"Output directory: {args.output_dir}")
    print(f"Total images: {args.total_images:,}")
    print(f"Update interval: {args.update_interval}s")
    print("=" * 60)

    try:
        while True:
            processed = count_processed_images(args.output_dir)
            running_jobs = get_running_jobs()
            progress_pct = (processed / args.total_images) * 100
            eta = estimate_completion_time(processed, args.total_images, start_time)

            elapsed_hours = (time.time() - start_time) / 3600

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Processed: {processed:,}/{args.total_images:,} ({progress_pct:.1f}%) | "
                  f"Running jobs: {running_jobs} | "
                  f"Elapsed: {elapsed_hours:.1f}h | "
                  f"ETA: {eta}")

            # Check if all jobs are done
            if running_jobs == 0 and processed > 0:
                print("\n" + "=" * 60)
                print("All jobs completed!")
                print(f"Final count: {processed:,} images processed")
                print(f"Total time: {elapsed_hours:.1f} hours")

                # Run final analysis
                if processed > 1000:  # Only if we have substantial data
                    print("Running final analysis...")
                    try:
                        subprocess.run([
                            'python', 'analyze_similarities.py',
                            '--predictions-dir', f'{args.output_dir}/predictions',
                            '--output', 'final_analysis_200k.json'
                        ])
                        print("Final analysis saved to: final_analysis_200k.json")
                    except Exception as e:
                        print(f"Could not run final analysis: {e}")

                break

            time.sleep(args.update_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        processed = count_processed_images(args.output_dir)
        print(f"Current progress: {processed:,} images processed")


if __name__ == "__main__":
    main()