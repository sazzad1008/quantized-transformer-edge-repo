# Quantized Transformer for Edge Inference

This repository contains:
- `quantasized_Transformer_Model_for_edge_device.ipynb`: original notebook workflow
- `train_quantized_transformer.py`: production-style training/evaluation/export script

## Features

- Automatic corpus bootstrap (`book.txt`) from Project Gutenberg if missing
- Character-level tiny Transformer language model training
- Optional fake quantization during training
- Dynamic INT8 quantized model export for CPU/edge inference
- JSON metrics output + checkpoint artifacts

## Quick Start

```bash
python train_quantized_transformer.py \
  --book-path book.txt \
  --epochs 3 \
  --batch-size 32 \
  --output-dir artifacts
```

## Common Options

```bash
python train_quantized_transformer.py \
  --book-path book.txt \
  --book-url https://www.gutenberg.org/files/1342/1342-0.txt \
  --epochs 5 \
  --batch-size 64 \
  --block-size 128 \
  --d-model 128 \
  --heads 4 \
  --layers 4 \
  --lr 3e-4 \
  --max-chars 200000 \
  --output-dir artifacts \
  --log-level INFO
```

## Outputs

By default, outputs are written to `artifacts/`:
- `best_fp32_model.pt`
- `best_int8_dynamic_model.pt`
- `metrics.json`

## Notes

- If CUDA is available, training uses GPU automatically. Otherwise it runs on CPU.
- For quick local tests, use `--max-chars` and fewer `--epochs`.
