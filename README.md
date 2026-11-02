# LLaVA Alignment

Preference alignment for vision-language models (VLMs). This repo has two parts:

1. **From-scratch DPO, KTO, and ORPO on BLIP-VQA** (`cs231nmilestone-main/`): custom loss implementations and a single training harness that compares them against a vanilla SFT baseline. This is the finished part of the project.
2. **Scaling to larger VLMs with TRL** (`idefics/`, `llava/`): TRL-based training scripts for Idefics2-8B and LLaVA-1.5-7B. This part is still in progress.

All of these methods skip the separate reward model and PPO loop of classic RLHF. They learn directly from preference data: a prompt and image, a preferred ("chosen") response, and a dispreferred ("rejected") one.

## Project status

| Component | Model | Methods | Status |
|---|---|---|---|
| `cs231nmilestone-main/` | `Salesforce/blip-vqa-base` | Vanilla SFT, DPO, KTO, ORPO | Complete, with experiment results |
| `idefics/` | `HuggingFaceM4/idefics2-8b` | DPO, CPO, KTO (via TRL) | DPO/CPO/KTO scripts written; `orpo.py` and `rloo.py` are empty stubs |
| `llava/` | `llava-hf/llava-1.5-7b-hf` | DPO (via TRL) | DPO launch config and notes only; the CPO/IPO/KTO notebooks are empty |

## Alignment methods

| Method | Idea | Reference model | Data | Paper |
|---|---|---|---|---|
| **DPO** | Reads an implicit reward off the policy/reference log-ratio and widens the chosen-vs-rejected gap | Yes (frozen) | Paired | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) |
| **KTO** | Prospect-theory loss; each example is scored against a KL reference point | Yes (frozen) | Unpaired good/bad labels | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) |
| **ORPO** | SFT loss plus an odds-ratio penalty, all in one stage | No | Paired | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) |
| **CPO** | Contrastive preference optimization; also reference-free | No | Paired | [Xu et al., 2024](https://arxiv.org/abs/2401.08417) |
| **IPO** | DPO variant with a regularized objective that resists overfitting | Yes (frozen) | Paired | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) |
| **RLOO** | REINFORCE Leave-One-Out, an online RL baseline | — | Reward signal | [Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740) |

The BLIP harness implements DPO, KTO, and ORPO. CPO, IPO, and RLOO only appear in the TRL-based scaling work, and only CPO has a script so far.

## Repository layout

```
llava_alignment/
├── cs231nmilestone-main/          # Part 1: from-scratch alignment on BLIP-VQA
│   ├── customblip.py              # ORPOBlipForVQA, PairBlipForVQA, DPO/KTO/ORPO losses
│   ├── customdataloaders.py       # VLFeedback + MLLM-DPO loaders, paired collation
│   ├── trainPA.py                 # Shared harness: 4 methods, LoRA, W&B, EM/F1/LLM-judge eval
│   ├── train-{vanilla,dpo,kto,orpo}.sh   # Configs used for the report runs
│   ├── train.ipynb                # Experiment log with saved outputs
│   └── requirements.txt
├── idefics/                       # Part 2: Idefics2-8B with TRL
│   ├── dpo.py                     # DPOTrainer on RLAIF-V, LoRA, bf16
│   ├── cpo.py                     # Same pipeline with CPOTrainer
│   ├── kto.py                     # Same pipeline with KTOTrainer
│   ├── orpo.py                    # (empty stub)
│   └── rloo.py                    # (empty stub)
├── llava/                         # Part 2: LLaVA-1.5 with TRL
│   ├── train-llava-dpo.sh         # accelerate launch config for TRL's DPO VLM example
│   ├── llava_dpo.ipynb            # Same config + reference links
│   ├── llava_orpo.ipynb           # Memory-budget notes
│   └── llava_{cpo,ipo,kto}.ipynb  # (empty)
├── TODO.md
└── README.md
```

## Installation

```bash
git clone https://github.com/catherinelee274/llava-alignment.git
cd llava-alignment

pip install -r cs231nmilestone-main/requirements.txt
pip install evaluate wandb trl   # used by the code but not yet in requirements.txt
```

`trainPA.py` uses LLaMA-3-8B on Together AI for its LLM-as-judge metric. Set your key before training:

```bash
export TOGETHER_API_KEY=...
```

## Part 1: BLIP-VQA (`cs231nmilestone-main/`)

### How it works

- **Custom model classes.** BLIP-VQA is an encoder-decoder, so TRL's decoder-only trainers don't apply. `ORPOBlipForVQA` and `PairBlipForVQA` override `forward()` so that the vision tower and question encoder run once, and the decoder then scores both the chosen and the rejected answer against the cached embeddings.
- **Sequence log-probs.** `_calculate_logps` shifts the labels for next-token prediction, gathers the per-token log-probs, masks out padding, and sums over the sequence.
- **Reference model.** DPO and KTO load a frozen copy of the model, initialized from the SFT checkpoint rather than the raw base, as DPO's derivation assumes.
- **One harness.** All four methods share the same dataloaders, eval loop, and logging. Only the `--alignment_method` flag changes.
- **Evaluation.** Exact match, token-level F1, and an LLM-as-judge score (LLaMA-3-8B through LangChain/Together, which grades each answer CORRECT or INCORRECT).

### Data

- **`MMInstruction/VLFeedback`**: loaded from the Hugging Face Hub.
- **`MLLM-DPO`**: a local JSON file plus an image directory from [findalexli/mllm-dpo](https://github.com/findalexli/mllm-dpo). The shell scripts expect `./data/dpo_inference0104.with_logpllava-v1.5-13b_2024-02-03.json` and the images under `./data/`.

### Training

Run vanilla SFT first. The DPO and KTO scripts start both the policy and the reference model from its checkpoint (`Model/blip-vanilla-mllm_vanilla`):

```bash
cd cs231nmilestone-main
bash train-vanilla.sh   # SFT baseline -> Model/blip-vanilla-mllm_vanilla
bash train-dpo.sh       # starts from the SFT checkpoint
bash train-kto.sh       # starts from the SFT checkpoint
bash train-orpo.sh      # reference-free; starts from the base model
```

Or call the harness directly:

```bash
python trainPA.py \
    --alignment_method dpo \               # vanilla | dpo | kto | orpo
    --pretrained_model Salesforce/blip-vqa-base \
    --save_model Model/blip-saved-model \
    --dataset MMInstruction/VLFeedback \   # or MLLM-DPO (+ --mllm-dpo-json, --mllm-dpo-images)
    --epochs 3 --batchsize 4 --lr 1e-5 \
    --train_percentage 10 --eval_percentage 5 \
    --lora True --lora_rank 16 --lora_alpha 16 \
    --wandb True --wandb_project cs231n_blip-vqa
```

| Flag | Default | Notes |
|---|---|---|
| `--alignment_method` | `vanilla` | `vanilla`, `dpo`, `kto`, `orpo` |
| `--epochs` / `--batchsize` / `--lr` | `1` / `2` / `1e-5` | |
| `--train_percentage` / `--eval_percentage` | `1` / `1` | Percentage of the split to use; `0` uses 10 records |
| `--max_eval_llm` | `0` | Max records per epoch scored by the LLM judge |
| `--dpo_beta` | `0.5` | |
| `--kto_beta` | `0.1` | |
| `--orpo_lambda` | `0.5` | |
| `--lora`, `--lora_rank`, `--lora_alpha` | `False`, `16`, `16` | |
| `--wandb`, `--wandb_project` | `False`, `cs231n_blip-vqa` | |
| `--push_to_hub` | `""` | Hub repo to push the trained model to |

> **Note:** `--lora`, `--wandb`, and `--use_scaler` are declared with `type=bool`, so any non-empty value, including `False`, turns them **on**. To keep one off, leave the flag out.

### Results

These runs used `blip-vqa-base` on MLLM-DPO, with sweeps over learning rate (1e-6 to 5e-5), LoRA rank (16/32/64), and epochs (1 to 10):

| Method | Train % | Epochs | F1 | EM | LLM judge |
|---|---|---|---|---|---|
| Vanilla SFT | 10 | 1 | **0.078** | 0.0 | 0.05 |
| ORPO | 10 | 1 | 0.040 | 0.0 | **0.125** |
| DPO | 10 | 1 | 0.009 | 0.0 | 0.10 |
| Vanilla SFT | 90 | 10 | 0.011 | 0.0 | 0.20 |

No alignment method beat the SFT baseline on F1 or EM. At matched settings, the LLM-judge scores rank ORPO > DPO > SFT, but the sample is small. Likely causes:

- **Model capacity.** `blip-vqa-base` (~385M params) was pretrained for short VQA answers, while MLLM-DPO contains long-form responses generated by LLaVA-13B. The model probably can't represent the preferred responses well enough for the preference signal to matter.
- **Metric mismatch.** EM is exactly 0.0 in every configuration, which points to a format mismatch between long generations and short references (or a post-processing bug), not to a real result. EM and F1 also don't measure hallucination, the behavior these methods target.

### Known issues

- **ORPO length normalization.** `_calculate_logps` returns *summed* log-probs, which is correct for DPO. ORPO's paper uses the length-normalized *mean*. With summed log-probs, `log(1 − exp(logp)) ≈ 0`, so the odds-ratio term degrades into a plain log-prob difference.
- **KTO on paired data.** The KTO loss is implemented, but it is fed paired chosen/rejected examples. That forfeits KTO's main advantage: learning from unpaired binary feedback.

## Part 2: Idefics2 and LLaVA with TRL (in progress)

### Idefics2-8B

`idefics/dpo.py` trains Idefics2-8B on `openbmb/RLAIF-V-Dataset` with TRL's `DPOTrainer`. It applies the chat template, uses LoRA on all linear layers, bf16, and gradient checkpointing, and runs with batch size 2 × grad accumulation 32. To avoid OOM, it shrinks each image to half the processor's maximum edge. `cpo.py` and `kto.py` run the same pipeline with `CPOTrainer` and `KTOTrainer`.

```bash
cd idefics
python dpo.py   # or cpo.py / kto.py
```

### LLaVA-1.5-7B

`llava/train-llava-dpo.sh` calls TRL's visual DPO example script on `HuggingFaceH4/rlaif-v_formatted`. Run it from a TRL repo checkout so that `examples/scripts/dpo_visual.py` resolves. The script path may differ in newer TRL releases.

```bash
accelerate launch examples/scripts/dpo_visual.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_llava_1_5 \
    --bf16 --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft --lora_target_modules=all-linear
```

**Memory budget.** For full fine-tuning with a reference model and AdamW, the model, reference, gradients, and two optimizer states take about 5 · N · P bytes (N = parameter count, P = bytes per parameter). This is why the scripts use LoRA, and why reference-free methods like ORPO and CPO are attractive at 7–8B scale.

## Next steps

1. Rerun the method comparison on a model that can represent the target responses (LLaVA-1.5-7B or Idefics2-8B).
2. Fix ORPO length normalization.
3. Evaluate on a hallucination benchmark (POPE, MMHal-Bench) instead of EM/F1.
4. Add an unpaired dataset path for KTO.
5. Fill in the ORPO and RLOO Idefics scripts and the remaining LLaVA notebooks.

## References

- [Hugging Face: Preference Optimization for Vision Language Models](https://huggingface.co/blog/dpo_vlm)
- [LLaVA: fine-tuning on custom data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)
- [findalexli/mllm-dpo](https://github.com/findalexli/mllm-dpo) and [alexshengzhili/llava-v1.5-13b-dpo](https://huggingface.co/alexshengzhili/llava-v1.5-13b-dpo)
- [RLAIF-V dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)
- [Unified Language Model Alignment](https://github.com/Unified-Language-Model-Alignment/src)

## Citation

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

@article{hong2024orpo,
  title={ORPO: Monolithic Preference Optimization without Reference Model},
  author={Hong, Jiwoo and Lee, Noah and Thorne, James},
  journal={arXiv preprint arXiv:2403.07691},
  year={2024}
}
```
