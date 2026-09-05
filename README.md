# LLaVA Alignment

A comprehensive repository for training Vision-Language Models (VLMs) with various preference alignment methods. This project implements multiple state-of-the-art alignment techniques for models including LLaVA, Idefics2, and BLIP.

## Overview

This repository provides implementations of several preference alignment techniques that improve Vision-Language Models without requiring traditional reinforcement learning. These methods align models with human preferences using paired preference data (chosen vs. rejected responses).

## Supported Models

- **LLaVA** (Large Language and Vision Assistant)
  - llava-1.5-7b-hf
  - llava-llama-3-8b
- **Idefics2** (Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS)
  - idefics2-8b
- **BLIP** (Bootstrapping Language-Image Pre-training)
  - blip-vqa-base

## Alignment Methods

The repository implements the following preference alignment techniques:

| Method | Description | Key Paper |
|--------|-------------|-----------|
| **DPO** | Direct Preference Optimization - Combines reward modeling and preference learning | Rafailov et al., 2023 |
| **KTO** | Kahneman-Tversky Optimization - Does not require pairwise preference data | Ethayarajh et al., 2024 |
| **CPO** | Conservative Preference Optimization - Conservative approach to preference alignment | - |
| **IPO** | Identity Preference Optimization - Prevents overfitting in DPO | Azar et al., 2023 |
| **ORPO** | Odds Ratio Preference Optimization - Merges SFT and preference alignment | Song et al., 2023 |
| **RLOO** | Reinforcement Learning from Language Output | - |

## Project Structure

```
llava_alignment/
├── llava/                      # LLaVA model implementations
│   ├── llava_dpo.ipynb        # DPO training notebook
│   ├── llava_kto.ipynb        # KTO training notebook
│   ├── llava_cpo.ipynb        # CPO training notebook
│   ├── llava_ipo.ipynb        # IPO training notebook
│   ├── llava_orpo.ipynb       # ORPO training notebook
│   └── train-llava-dpo.sh     # Shell script for DPO training
├── idefics/                    # Idefics2 model implementations
│   ├── dpo.py                 # DPO training script
│   ├── kto.py                 # KTO training script
│   ├── cpo.py                 # CPO training script
│   ├── orpo.py                # ORPO training script
│   └── rloo.py                # RLOO training script
├── cs231nmilestone-main/       # BLIP model custom implementations
│   ├── trainPA.py             # Main training script with multiple methods
│   ├── customblip.py          # Custom BLIP implementations
│   ├── customdataloaders.py   # Custom data loaders
│   ├── train-dpo.sh           # DPO training shell script
│   ├── train-kto.sh           # KTO training shell script
│   ├── train-orpo.sh          # ORPO training shell script
│   └── train-vanilla.sh       # Vanilla SFT training shell script
├── blip2/                      # BLIP2 implementations
├── llava-llama3/              # LLaVA-Llama3 implementations
├── qwen/                       # Qwen model implementations
└── README.md                   # This file
```

## Datasets

The project supports multiple preference datasets:

- **HuggingFaceH4/rlaif-v_formatted**: Pre-formatted RLAIF-V dataset for visual instruction tuning
- **openbmb/RLAIF-V-Dataset**: Original RLAIF-V dataset with preference annotations
- **MMInstruction/VLFeedback**: Visual feedback dataset for VLM alignment
- **MLLM-DPO**: Custom multi-modal language model DPO dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/catherinelee274/llava-alignment.git
cd llava-alignment

# Install dependencies
pip install -r cs231nmilestone-main/requirements.txt

# Main dependencies include:
# - transformers
# - trl (Transformer Reinforcement Learning)
# - peft (Parameter-Efficient Fine-Tuning)
# - accelerate
# - datasets
# - torch
# - torchvision
```

## Quick Start

### Training LLaVA with DPO

Using the shell script:
```bash
cd llava
bash train-llava-dpo.sh
```

Or using accelerate directly:
```bash
accelerate launch examples/scripts/dpo_visual.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_llava_1_5 \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules=all-linear
```

### Training Idefics2 with DPO

```bash
cd idefics
python dpo.py
```

### Training BLIP with Multiple Methods

The BLIP implementation supports multiple alignment methods:

```bash
cd cs231nmilestone-main

# Train with DPO
python trainPA.py --alignment_method dpo --epochs 3 --batchsize 4 --lr 1e-5

# Train with KTO
python trainPA.py --alignment_method kto --epochs 3 --batchsize 4 --lr 1e-5

# Train with ORPO
python trainPA.py --alignment_method orpo --epochs 3 --batchsize 4 --lr 1e-5

# Train with vanilla SFT
python trainPA.py --alignment_method vanilla --epochs 3 --batchsize 4 --lr 1e-5
```

## Training Options

### BLIP Training Script Arguments

```bash
python trainPA.py \
    --alignment_method dpo \              # dpo, kto, orpo, or vanilla
    --pretrained_model Salesforce/blip-vqa-base \
    --save_model Model/blip-saved-model \
    --epochs 3 \
    --batchsize 4 \
    --lr 1e-5 \
    --train_percentage 10 \               # Use 10% of training data
    --eval_percentage 5 \                 # Use 5% of eval data
    --dataset MMInstruction/VLFeedback \  # Dataset to use
    --lora True \                         # Use LoRA for efficient training
    --lora_rank 16 \
    --lora_alpha 16 \
    --wandb True \                        # Enable W&B logging
    --wandb_project cs231n_blip-vqa
```

### Hyperparameters

**DPO-specific:**
- `--dpo_beta`: Beta value for DPO loss (default: 0.5)

**KTO-specific:**
- `--kto_beta`: Beta value for KTO loss (default: 0.1)

**ORPO-specific:**
- `--orpo_lambda`: Lambda value for ORPO loss (default: 0.5)

**LoRA/PEFT:**
- `--lora`: Enable LoRA for parameter-efficient training
- `--lora_rank`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling factor (default: 16)

## Features

### Training Features

- ✅ **Multiple Alignment Methods**: DPO, KTO, CPO, IPO, ORPO, RLOO
- ✅ **Parameter-Efficient Training**: LoRA/PEFT support for all models
- ✅ **Mixed Precision Training**: bf16 and fp16 support
- ✅ **Gradient Checkpointing**: Memory-efficient training
- ✅ **Multi-GPU Support**: Accelerate integration for distributed training
- ✅ **Custom Implementations**: Custom BLIP models with preference alignment

### Evaluation Features

- ✅ **Exact Match (EM)**: Token-level accuracy
- ✅ **F1 Score**: Word-level overlap metric
- ✅ **LLM-based Evaluation**: Using LLaMA-3 for qualitative assessment
- ✅ **Loss Tracking**: Training and validation loss monitoring
- ✅ **W&B Integration**: Experiment tracking with Weights & Biases

## Model Architecture Details

### LLaVA
- Vision encoder + Language model architecture
- Supports various sizes (7B, 13B)
- Cross-modal alignment via projection layer

### Idefics2
- Interleaved vision-language architecture
- 8B parameter model
- Image splitting and processing capabilities

### BLIP
- Vision-Language Pre-training architecture
- Question answering capabilities
- Custom implementations for preference alignment

## Key Implementations

### Custom BLIP Models

Located in `cs231nmilestone-main/customblip.py`:

- `ORPOBlipForVQA`: BLIP model with ORPO loss
- `PairBlipForVQA`: BLIP model with DPO/KTO support
- Custom loss calculations for each alignment method
- Reference model support for DPO/KTO

### Data Loaders

Located in `cs231nmilestone-main/customdataloaders.py`:

- VLFeedback dataset loader
- MLLM-DPO dataset loader
- Preprocessing and tokenization
- Batch collation functions

## Performance Metrics

Based on the TODO notes, the performance hierarchy observed:

```
KTO > DPO ≈ CPO > IPO, MDPO
```

This suggests that KTO (which doesn't require paired preferences) performs best, followed by DPO and CPO with similar performance.

## Related Work & References

- [Hugging Face DPO VLM Blog](https://huggingface.co/blog/dpo_vlm)
- [LLaVA Fine-tuning Documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)
- [Pre-trained DPO LLaVA Model](https://huggingface.co/alexshengzhili/llava-v1.5-13b-dpo)
- [MLLM-DPO Implementation](https://github.com/findalexli/mllm-dpo)
- [Unified Language Model Alignment](https://github.com/Unified-Language-Model-Alignment/src)

## Hardware Requirements

**Minimum Requirements:**
- GPU: NVIDIA GPU with 16GB VRAM (e.g., V100, A10)
- RAM: 32GB system memory
- Storage: 50GB for models and datasets

**Recommended:**
- GPU: NVIDIA A100 (40GB/80GB) or H100
- RAM: 64GB+ system memory
- Multi-GPU setup for faster training

## Tips for Training

1. **Memory Management**:
   - Use gradient checkpointing for larger models
   - Enable LoRA/PEFT to reduce memory footprint
   - Adjust batch size and gradient accumulation steps

2. **Hyperparameter Tuning**:
   - Start with default beta/lambda values
   - Lower learning rates (1e-5 to 5e-6) work well
   - Monitor both training and eval losses

3. **Dataset Considerations**:
   - Ensure balanced chosen/rejected pairs
   - Quality of preference data matters more than quantity
   - Consider data augmentation for small datasets

## Citation

If you use this repository in your research, please cite the relevant papers:

```bibtex
@article{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}

@article{ethayarajh2024kto,
  title={KTO: Model Alignment as Prospect Theoretic Optimization},
  author={Ethayarajh, Kawin and Xu, Winnie and Muennighoff, Niklas and Jurafsky, Dan and Kiela, Douwe},
  journal={arXiv preprint arXiv:2402.01306},
  year={2024}
}
```