# Segmentation Integration Testing Plan

## Testing Phases

### Phase 1: Component Unit Testing
- **1.1 VAE Segmentation Methods**
  - Test `encode_segmentation_mask()` with various mask shapes
  - Test `decode_segmentation_latent()` reconstruction quality
  - Test `encode_joint()` consistency between image and mask encoding
  - Validate shape compatibility and data type handling

- **1.2 DiT Segmentation Head**
  - Test segmentation prediction output shapes and ranges
  - Validate dual output (image noise + segmentation mask)
  - Test with and without segmentation enabled
  - Check gradient flow through segmentation head

- **1.3 Dataset Integration**
  - Test `UnifiedDatasetSegmentation` mask loading
  - Validate RLE mask parsing and tensor conversion
  - Test error handling for missing masks
  - Check mask preprocessing and resizing

- **1.4 Pipeline Units**
  - Test `QwenImageUnit_SegmentationMaskProcessor`
  - Validate mask encoding through pipeline
  - Test integration with other pipeline units

### Phase 2: Integration Testing
- **2.1 Training Loss Function**
  - Test combined loss calculation (image MSE + segmentation BCE)
  - Validate gradient computation for both losses
  - Test loss weighting with different `mask_loss_weight` values
  - Check loss stability and convergence

- **2.2 End-to-End Training Pipeline**
  - Test full training loop with segmentation
  - Validate data flow from dataset → pipeline → loss → backward
  - Test checkpoint saving/loading with segmentation models
  - Check memory usage and computational efficiency

### Phase 3: Practical Validation
- **3.1 Small-Scale Training**
  - Run training on synthetic dataset
  - Monitor loss convergence
  - Validate spatial learning effectiveness
  - Check for runtime errors or crashes

- **3.2 Generation Testing**
  - Test model inference with spatial awareness
  - Compare results with/without segmentation training
  - Validate that model learned spatial boundaries
  - Test different `mask_loss_weight` configurations

### Phase 4: Performance and Optimization
- **4.1 Memory Usage**
  - Monitor VRAM usage during training
  - Test batch size limitations
  - Check for memory leaks

- **4.2 Computational Efficiency**
  - Measure training speed impact
  - Test inference time overhead
  - Optimize where needed

## Expected Issues to Investigate

### Potential Bug Categories
1. **Shape Mismatches**: Tensor shapes not aligning between components
2. **Data Type Issues**: Inconsistent dtype handling (float32, float16, bfloat16)
3. **Memory Leaks**: Unreleased GPU memory or CPU memory
4. **Gradient Issues**: Broken gradient flow through segmentation head
5. **Integration Errors**: Pipeline unit conflicts or data flow problems
6. **Training Instability**: Loss exploding/vanishing, NaN values
7. **Performance Issues**: Slow training/inference due to inefficient operations

### Test Data Requirements
- Synthetic images with known segmentation patterns
- Real image data with corresponding masks
- Edge cases: empty masks, full masks, irregular shapes
- Various resolutions: 64x64, 128x128, 256x256, 512x512

## Success Criteria

### Functional Requirements
- [ ] All unit tests pass without errors
- [ ] Integration tests succeed with reasonable metrics
- [ ] Small-scale training converges without crashes
- [ ] Generated images show spatial awareness improvements
- [ ] No memory leaks or performance regressions

### Quality Metrics
- [ ] Segmentation loss decreases during training
- [ ] Generated edits respect spatial boundaries
- [ ] Training loss stabilizes and converges
- [ ] Model produces consistent spatial guidance

## Implementation Priority

1. **High Priority**: Unit tests for core components
2. **Medium Priority**: Integration tests and basic training
3. **Low Priority**: Performance optimization (after functional validation)

## Testing Timeline

- **Day 1**: Component unit testing and bug fixing
- **Day 2**: Integration testing and training validation
- **Day 3**: Practical validation and optimization

This plan ensures comprehensive coverage while focusing on functional correctness first, then performance optimization.