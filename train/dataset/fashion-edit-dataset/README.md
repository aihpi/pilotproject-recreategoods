---
license: apache-2.0
dataset_info:
  features:
  - name: input_image
    dtype: binary
  - name: output_image
    dtype: binary
  - name: original_caption
    dtype: string
  - name: edit_instruction
    dtype: string
  - name: resulting_caption
    dtype: string
  - name: class_name
    dtype: string
  - name: status
    dtype: string
  splits:
  - name: val
    num_bytes: 24027562899
    num_examples: 17493
  - name: test
    num_bytes: 24541391419
    num_examples: 17804
  download_size: 48553446847
  dataset_size: 48568954318
configs:
- config_name: default
  data_files:
  - split: train
    path: train/train-*
  - split: val
    path: val/val-*
  - split: test
    path: test/test-*
---
