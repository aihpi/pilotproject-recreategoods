#!/bin/bash
# check_slurm_logs.sh - Comprehensive diagnostic script for Slurm job failures

JOB_ID="1395923"
PROJECT_DIR="/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune"

echo "=== Slurm Job Diagnostic Report ==="
echo "Job ID: $JOB_ID"
echo "Timestamp: $(date)"
echo

echo "1. Check job status:"
sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS,CPUTime,AllocCPUS,AllocNodes
echo

echo "2. Detailed job information:"
sacct -j $JOB_ID --long
echo

echo "3. Check standard output log:"
if [ -f "$PROJECT_DIR/logs/train_${JOB_ID}.out" ]; then
    echo "=== STANDARD OUTPUT ==="
    tail -50 "$PROJECT_DIR/logs/train_${JOB_ID}.out"
else
    echo "No output log found: $PROJECT_DIR/logs/train_${JOB_ID}.out"
fi
echo

echo "4. Check error log:"
if [ -f "$PROJECT_DIR/logs/train_${JOB_ID}.err" ]; then
    echo "=== STANDARD ERROR ==="
    tail -50 "$PROJECT_DIR/logs/train_${JOB_ID}.err"
else
    echo "No error log found: $PROJECT_DIR/logs/train_${JOB_ID}.err"
fi
echo

echo "5. Check recent Slurm messages:"
squeue -j $JOB_ID -h | head -5
echo

echo "6. Check partition availability:"
sinfo -p aisc | head -10
echo

echo "7. Check for similar job failures in recent history:"
sacct --starttime="$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S')" --format=JobID,JobName,State,ExitCode | grep -E "launch failed|failed|Error"
echo

echo "8. Environment check:"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo

echo "9. Check critical files exist:"
echo "Training script exists: $([ -f "$PROJECT_DIR/DiffSynth-Studio/examples/qwen_image/model_training/train_with_segmentation.py" ] && echo "YES" || echo "NO")"
echo "Dataset directory exists: $([ -d "$PROJECT_DIR/data/example_image_dataset" ] && echo "YES" || echo "NO")"
echo "Logs directory exists: $([ -d "$PROJECT_DIR/logs" ] && echo "YES" || echo "NO")"
echo "Models directory exists: $([ -d "$PROJECT_DIR/models/train" ] && echo "YES" || echo "NO")"
echo

echo "10. Quick memory and GPU availability check:"
free -h
nvidia-smi 2>/dev/null | grep -E "memory|GTX|RTX|A100|H100" || echo "nvidia-smi not available or no GPUs found"
echo

