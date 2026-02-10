# Curriculum Training Debug Session Summary

## Overview
This document summarizes the debugging session for QwenImage curriculum training, including all issues found, solutions implemented, and system optimizations.

## Initial Problem
- Curriculum training stopping at 15 epochs instead of continuing properly
- IndexError crashes during training related to scheduler timesteps
- Epochs completing too quickly (10 seconds each)
- Poor curriculum training results compared to baseline

## Issues Found & Solutions Implemented

### 1. **Scheduler State Corruption During Validation**
**Problem**: IndexError "index 722 is out of bounds for dimension 0 with size 20"
- Cause: Scheduler timesteps getting corrupted during validation inference
- Location: `DiffSynth-Studio/diffsynth/trainers/utils.py`

**Solution**: Implemented scheduler state preservation
```python
def _inference_single_image(self, model, inputs):
    # Save scheduler state before inference
    if hasattr(model, 'module'):
        original_timesteps = model.module.pipe.scheduler.timesteps.clone()
        model.module.pipe.scheduler.set_timesteps(40, training=False)
    else:
        original_timesteps = model.pipe.scheduler.timesteps.clone()
        model.pipe.scheduler.set_timesteps(40, training=False)

    # ... inference code ...

    # Restore scheduler state after inference
    if hasattr(model, 'module'):
        model.module.pipe.scheduler.timesteps = original_timesteps
        model.module.pipe.scheduler.set_timesteps(1000, training=True)
    else:
        model.pipe.scheduler.timesteps = original_timesteps
        model.pipe.scheduler.set_timesteps(1000, training=True)
```

### 2. **Type Annotation Errors**
**Problem**: PipelineUnitRunner return type mismatch
- Error: Expected `tuple[dict, dict]` but returns `tuple[dict, dict, dict]`
- Location: `DiffSynth-Studio/diffsynth/utils/__init__.py`

**Solution**: Fixed type annotation
```python
PipelineUnitRunner = Callable[[Any, Any, dict, dict, dict], tuple[dict, dict, dict]]
```

### 3. **Epoch Duration Too Short**
**Problem**: Curriculum epochs completing in 10 seconds
- Original: `curriculum_steps_per_epoch = 100` → 500 samples per epoch
- User reported first real gains at 64,000 images

**Solution**: Increased epoch size dramatically
```bash
--curriculum_steps_per_epoch 12800    # 64,000 samples per epoch (5 GPUs × 12,800 steps)
--validation_steps 12800              # Validate once per epoch
--save_steps 6400                     # Save twice per epoch
```

### 4. **Inefficient Validation (Single GPU Usage)**
**Problem**: Only rank 0 GPU doing validation, others idle
- 20 validation samples processed sequentially on 1 GPU
- Other 4 GPUs sitting idle during validation

**Solution**: Implemented distributed validation
```python
def evaluate_batch(self, model, validation_samples, batch_size=4, save_images=False, epoch=None, max_save=3, accelerator=None):
    # Distribute validation samples across GPUs
    if accelerator is not None and accelerator.num_processes > 1:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        # Distribute samples round-robin across ranks
        my_samples = [validation_samples[i] for i in range(rank, len(validation_samples), world_size)]
    else:
        my_samples = validation_samples

    # ... process samples ...

    # Gather results from all GPUs
    if accelerator is not None and accelerator.num_processes > 1:
        local_avg_tensor = torch.tensor(local_avg_lpips, device=accelerator.device)
        gathered_lpips = accelerator.gather(local_avg_tensor)
        if accelerator.is_main_process:
            valid_scores = [score.item() for score in gathered_lpips if score.item() != float('inf')]
            final_avg_lpips = sum(valid_scores) / len(valid_scores) if valid_scores else float('inf')
            return final_avg_lpips
    else:
        return local_avg_lpips
```

### 5. **Batch Size Distribution Constraints**
**Problem**: Unnecessary assertion requiring batch_size divisible by world_size
- Location: `DiffSynth-Studio/diffsynth/curriculum.py`
- Error: "AssertionError: batch_size must be divisible by world_size"

**Solution**: Removed constraint, implemented sample-level sharding
```python
# Note: Removed divisibility constraint - sample-level sharding works with any batch size
self.world_size = int(world_size)
self.rank = int(rank)
```

### 6. **Account and Fairshare Optimization**
**Problem**: Using aisc-staff account with lower fairshare
- aisc-staff: 46.09% fairshare (3.6B usage)
- aisc: 47.17% fairshare (519M usage) - BETTER

**Solution**: Switched to aisc account for better scheduling priority
```bash
#SBATCH --account=aisc        # Better fairshare
#SBATCH --qos=aisc
```

## Current Configuration

### Curriculum Training Parameters
```bash
#!/bin/bash
#SBATCH --job-name=qwen-curriculum-test
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# Curriculum parameters
--curriculum_steps_per_epoch 12800     # 64,000 samples per epoch
--validation_steps 12800               # Validate once per epoch
--validation_batch_size 4              # 4 samples per GPU (distributed)
--validation_samples 20                # Total validation samples
--validation_patience 2                # Phase advancement patience
--validation_min_delta 0.005          # LPIPS improvement threshold
--max_epochs_per_phase 4              # Max epochs before forced advancement
--save_steps 6400                     # Save twice per epoch
```

### Adaptive Curriculum Phases
```python
phases = [
    {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},   # Phase 1: Easy only
    {0: 0.7, 1: 0.3, 2: 0.0, 3: 0.0},   # Phase 2: Mostly easy + some medium
    {0: 0.4, 1: 0.4, 2: 0.2, 3: 0.0},   # Phase 3: Easy + medium + some hard
    {0: 0.2, 1: 0.4, 2: 0.4, 3: 0.0},   # Phase 4: Medium + hard focus
    {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.3},   # Phase 5: All difficulties
]
```

## Fairshare Management Strategy

### Current Status
- **aisc account**: 47.17% fairshare (519M usage) - PREFERRED
- **aisc-staff account**: 46.09% fairshare (3.6B usage)

### Recovery Timeline (14-day half-life)
- **2 weeks**: ~50-55% fairshare recovery
- **1 month**: ~60-70% fairshare recovery
- **2 months**: ~80-90% fairshare recovery

### Strategic Usage
```bash
# Use aisc for bulk/routine work
--account=aisc

# Reserve aisc-staff for urgent/critical work
--account=aisc-staff
```

### Resource Impact on Fairshare
```
PriorityWeightTRES = CPU=1000,Mem=4000,GRES/gpu=5000

GPU jobs: 5x impact vs CPU
Memory: 4x impact vs CPU
CPU: Base impact
```

### VSCode Remote Impact
- Current VSCode session: 12 CPUs + 128GB RAM
- Daily fairshare cost: ~500K units
- **Recommendation**: Reduce to 4 CPUs + 32GB

## Validation System

### Distributed Validation Benefits
- **Before**: 1 GPU processes 20 samples sequentially
- **After**: 5 GPUs process 4 samples each in parallel
- **Speedup**: ~5x validation performance
- **Resource distribution**:
  - GPU 0: samples [0, 5, 10, 15]
  - GPU 1: samples [1, 6, 11, 16]
  - GPU 2: samples [2, 7, 12, 17]
  - GPU 3: samples [3, 8, 13, 18]
  - GPU 4: samples [4, 9, 14, 19]

### Validation Settings
- **LPIPS evaluation**: 20 samples per validation run
- **Image saving**: Only 2 images saved (rank 0 only)
- **Inference steps**: 40 (high quality validation)
- **Frequency**: Once per epoch (12,800 steps)

## Key Files Modified

### 1. `DiffSynth-Studio/diffsynth/trainers/utils.py`
- Added scheduler state preservation in `_inference_single_image()`
- Implemented distributed validation in `LPIPSEvaluator.evaluate_batch()`
- Fixed validation result gathering with `accelerator.gather()`
- Added rank 0 checks for clean output

### 2. `DiffSynth-Studio/diffsynth/curriculum.py`
- Removed unnecessary batch size constraint in `DistributedCurriculumSampler`
- Added note about sample-level sharding

### 3. `DiffSynth-Studio/diffsynth/utils/__init__.py`
- Fixed PipelineUnitRunner type annotation

### 4. `test_curriculum.sbatch`
- Updated resource allocation and timing parameters
- Switched to aisc account for better fairshare

## Remaining Considerations

### 1. Fundamental Curriculum vs Regular Training Difference
- Issue: Curriculum training produces different results than regular training
- Status: Partially addressed by fixing epoch duration and validation
- Next: May need deeper investigation into data flow differences

### 2. Memory Optimization
- Current: Using simpler accelerator setup to avoid CUDA OOM
- Alternative: Could investigate more sophisticated DDP approaches

### 3. Scheduler Optimization
- Current: Using gradual curriculum schedule
- Alternative: Could experiment with different phase transition strategies

## Monitoring Commands

### Check Job Status
```bash
squeue -u $USER
sprio -u $USER                    # Check priority
sshare -u $USER                   # Check fairshare
```

### Validation Monitoring
```bash
tail -f logs/curriculum_test_*.out
```

### Resource Usage
```bash
sstat -j <jobid> --format=JobID,MaxVMSize,MaxRSS,MaxPages,AveCPU
```

## Success Criteria
- [x] Eliminated IndexError crashes
- [x] Implemented distributed validation
- [x] Fixed epoch duration (64K samples per epoch)
- [x] Optimized fairshare usage
- [x] Clean validation output with proper image saving
- [ ] Verify curriculum produces better results than baseline

---

**Session Date**: 2025-09-18
**Status**: Major issues resolved, curriculum training ready for testing