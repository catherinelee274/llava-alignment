# LLaVA-Alignment — Interview Notes

Preference alignment for vision-language models. Custom from-scratch implementations of
DPO, KTO, and ORPO on BLIP-VQA, benchmarked against vanilla SFT, plus a partial port of the
comparison to LLaVA-1.5 and Idefics2 using TRL.

---

## 1. The 30-second pitch

> I implemented three preference-alignment objectives — DPO, KTO, and ORPO — from scratch
> on a vision-language model, rather than calling a library trainer. That meant subclassing
> BLIP's VQA head to run chosen and rejected answers through the decoder in a single forward
> pass, computing masked sequence log-probabilities by hand, and managing a frozen reference
> model for the methods that need one. I built a shared harness so all four methods (including
> a vanilla SFT baseline) run through one training loop with identical data and evaluation,
> then benchmarked them on visual preference data with exact-match, F1, and an LLM-as-judge
> score. The headline result was negative — the base model was too small for the methods to
> separate — which turned out to be the more interesting thing to talk about.

## 2. The 2-minute version

**Motivation.** Classic RLHF requires training a separate reward model and then running PPO
against it: three models in memory, unstable, expensive. A wave of 2023–24 papers showed you
can skip the reward model entirely and optimize preferences directly. I wanted to know whether
those results — all demonstrated on text-only LLMs — transfer to *vision*-language models,
where the failure mode you're correcting isn't tone or helpfulness but **hallucination**:
the model confidently describing objects that aren't in the image.

**Approach.** Rather than use TRL's `DPOTrainer` as a black box, I implemented the losses
directly so I actually understood the mechanics. Three custom model classes, one shared
training/eval harness, four methods behind a single `--alignment_method` flag so the
comparison is apples-to-apples.

**Outcome.** All methods trained stably and the losses behaved as the papers predict, but
downstream metrics never separated from the baseline. Diagnosing *why* — model capacity,
metric mismatch, and a length-normalization bug I found later — was the real work.

---

## 3. What each method actually does

All three take a prompt `x`, a preferred response `y_w`, and a rejected response `y_l`.

### DPO — Direct Preference Optimization (Rafailov et al., 2023)

The insight: the optimal RLHF policy has a closed-form relationship to the reward function,
so you can invert it and read the reward *implicitly* off your own model. The implicit reward
of a response is how much more your policy likes it than a frozen reference copy does.

```
r(x,y) = β · [ log π_θ(y|x) − log π_ref(y|x) ]

L_DPO = −log σ( β · [ (logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l)) ] )
```

In plain terms: widen the gap between chosen and rejected *relative to where you started*.
The frozen reference is the leash that stops the policy drifting into degenerate text;
`β` sets its length (higher β = tighter constraint).

- **Needs:** a frozen reference model (2× memory) and strictly *paired* preference data.
- **My implementation:** `customblip.py::calculate_dpo_loss`

### KTO — Kahneman-Tversky Optimization (Ethayarajh et al., 2024)

Framed via prospect theory: humans weigh losses more heavily than equivalent gains and
evaluate outcomes against a reference point. The theory motivates the loss shape, but the
practical payoff is unrelated — **KTO drops the pairing requirement**. Each example only needs
a binary desirable/undesirable label, and thumbs-up/thumbs-down feedback is far cheaper to
collect at scale than "rank these two responses."

Same implicit reward `r_θ = logπ_θ − logπ_ref`, but each example is scored against `z₀`,
a KL-divergence estimate serving as the reference point:

```
desirable:    v = σ( β · (r_θ − z₀) )
undesirable:  v = σ( β · (z₀ − r_θ) )
L_KTO = E[ λ − v ]
```

`z₀` is computed under `no_grad()` — the paper notes backpropagating through it is unstable.

- **My implementation:** `customblip.py::calculate_kto_loss`, with `_z0()` and `_r_t()` helpers.

### ORPO — Odds Ratio Preference Optimization (Hong et al., 2024)

Eliminates the reference model *entirely* and collapses SFT and alignment into a single stage.
Instead of a ratio against a reference, it appends an odds-ratio penalty to ordinary
cross-entropy:

```
odds(y|x) = p(y|x) / (1 − p(y|x))

L_ORPO = L_SFT − λ · log σ( log[ odds(y_w|x) / odds(y_l|x) ] )
```

Do normal supervised finetuning on the good answer, plus a term pushing its odds above the
bad one's. Halves memory (no reference model) and removes a whole training stage.

- **My implementation:** `customblip.py::calculate_orpo_loss`

### Comparison

| | Reference model | Data format | Pipeline stages | Memory |
|---|---|---|---|---|
| SFT (baseline) | No | Chosen only | 1 | 1× |
| **DPO** | Yes (frozen) | Paired | SFT → DPO | 2× |
| **KTO** | Yes (frozen) | Unpaired binary | SFT → KTO | 2× |
| **ORPO** | **No** | Paired | 1 | 1× |

---

## 4. Implementation — what I actually wrote

### Custom model classes (`cs231nmilestone-main/customblip.py`)

BLIP's VQA head is an encoder-decoder: a ViT encodes the image, a text encoder fuses the
question with image embeddings via cross-attention, and a text decoder generates the answer.
The stock class returns one cross-entropy loss for one answer — preference methods need
**two** answers scored per example.

- **`ORPOBlipForVQA`** — overrides `forward()` to run the decoder twice against the same
  cached vision/question embeddings (once with chosen labels, once with rejected). The vision
  tower and text encoder only run once per batch, so the marginal cost of the second answer
  is just the decoder.
- **`PairBlipForVQA`** — same dual-decode structure, but returns raw logits for both branches
  so DPO and KTO can compute their own log-probs. Shared by both methods; the config object
  passed at construction picks the loss.

**Sequence log-probs (`_calculate_logps`)** — the piece worth being able to explain on a
whiteboard. Both DPO and KTO need `log π(y|x)`, the log-probability of an entire answer:

```python
labels = labels[:, 1:].clone()        # shift for next-token prediction
logits = logits[:, :-1, :]
loss_mask = (labels != -100)          # ignore padding / prompt positions
labels[labels == -100] = 0            # gather() can't index with -100
per_token_logps = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)
return (per_token_logps * loss_mask).sum(-1)
```

Shift-by-one for next-token alignment, log-softmax over vocab, gather the log-prob of the
*actual* token at each position, zero out padding, sum along the sequence.

### Reference model handling (`trainPA.py`)

DPO and KTO load a second copy of the model, freeze it (`requires_grad = False`), and run it
under `torch.no_grad()`. Crucially the reference is initialized from the **SFT checkpoint**,
not the raw pretrained base — my shell scripts run vanilla SFT first
(`Model/blip-vanilla-mllm_vanilla`) and pass that as the starting point for both policy and
reference. This is correct practice: DPO's derivation assumes the reference is the SFT policy.

### Shared harness

One `calculate_loss()` dispatch means all four methods share the same dataloaders, eval loop,
logging, and checkpoint selection — the only variable is the objective. Also supports
LoRA/PEFT, W&B logging, and configurable train/eval data fractions for fast iteration.

### Data (`customdataloaders.py`)

Two preference sources behind one `VQADataset`: **VLFeedback** (HF hub) and **MLLM-DPO**
(local JSON + image directory). The collator tokenizes chosen and rejected answers into
`labels` and `neg_labels` with matched padding, and processes images through `BlipProcessor`.

### Evaluation

Three complementary metrics, because no single one is trustworthy:
- **Exact match** — strict, appropriate for short VQA answers
- **Token-level F1** — partial credit for overlapping words
- **LLM-as-judge** — LLaMA-3 via LangChain/Together grading answers CORRECT/INCORRECT,
  to catch semantically-right-but-lexically-different answers

---

## 5. Results — and how to talk about them

Ran on `Salesforce/blip-vqa-base` with the MLLM-DPO dataset, sweeping learning rates
(1e-6 → 5e-5), LoRA ranks (16/32/64), and epochs (1–10).

| Method | Train % | Epochs | F1 | EM | LLM judge |
|---|---|---|---|---|---|
| Vanilla SFT | 10 | 1 | **0.078** | 0.0 | 0.05 |
| ORPO | 10 | 1 | 0.040 | 0.0 | 0.125 |
| DPO | 10 | 1 | 0.009 | 0.0 | 0.10 |
| Vanilla SFT | 90 | 10 | 0.011 | 0.0 | 0.20 |

**Honest read: no method beat the baseline, and exact match was 0.0 everywhere.**

Don't hide this in an interview — lead with the diagnosis, which is the part that
demonstrates judgment:

1. **Model capacity.** `blip-vqa-base` is ~385M parameters and pretrained for *short* VQA
   answers ("yes", "two", "red"). The MLLM-DPO preference data is long-form descriptive text
   generated by LLaVA-13B. The base model likely cannot represent the preferred responses at
   all, so there's no meaningful preference signal to optimize — the methods can't separate
   because the model can't express the difference.
2. **Metric mismatch.** EM of exactly 0.0 across every configuration is a red flag, not a
   result. It strongly suggests a format mismatch between generated long-form output and
   short reference answers, or a decoding/post-processing bug — not that the model learned
   nothing. F1 near 0.01–0.08 is consistent with that.
3. **The alignment tax is real but I couldn't measure it.** Preference methods trade
   task-metric performance for behavioral quality (less hallucination, better calibration).
   EM/F1 are exactly the wrong instruments for that — which is why I added the LLM judge, and
   notably the LLM-judge scores *do* order roughly as expected (ORPO 0.125 > DPO 0.10 >
   SFT 0.05 at matched settings). Small sample, so I wouldn't over-claim it.

### Known bug I'd fix first

`_calculate_logps` returns the **sum** of per-token log-probs. That's correct for DPO — the
paper defines it that way. But **ORPO's paper uses the length-normalized mean**, and my ORPO
loss reuses the summed version. Because a summed sequence log-prob is a large negative number,
`exp(pos_prob) ≈ 0`, so `log(1 − exp(pos_prob)) ≈ 0` and the odds-ratio term silently degrades
to a plain log-prob difference — the ORPO objective collapses toward DPO-without-a-reference.
TRL's `ORPOTrainer` uses the average log-prob for exactly this reason. This alone could explain
ORPO's underperformance.

Being able to say *"here's a real bug in my own code, here's the mechanism, here's how I'd
verify the fix"* is worth more in an interview than a clean result.

---

## 6. Part 2 — scaling to real VLMs (partial)

The natural follow-up: rerun the comparison on models large enough for the alignment signal to
matter. This stage is scaffolded but incomplete, so describe it as *in-progress*, not shipped.

- `idefics/dpo.py` — full TRL `DPOTrainer` pipeline for **Idefics2-8B** on `openbmb/RLAIF-V-Dataset`:
  chat-template formatting, LoRA on all linear layers, bf16, gradient checkpointing,
  batch size 2 × grad accum 32. Includes an image-resize step (`thumbnail()` to half the
  processor's max edge) to avoid OOM — a practical detail specific to VLM training.
- `llava/` — `accelerate launch dpo_visual.py` configs for **llava-1.5-7b** on
  `rlaif-v_formatted`, plus memory-budget notes (model + reference + gradients + 2× AdamW
  states = 5·N·P, which is the whole argument for reference-free ORPO and for LoRA).
- `cpo.py` / `kto.py` variants exist; ORPO and RLOO for Idefics are stubs.

**Why RLAIF-V matters here:** it's built specifically for VLM hallucination — the rejected
responses describe things not present in the image. That makes it the right benchmark for the
question I was actually asking.

---

## 7. VLM vs. LLM vs. image-generation finetuning

A likely interview question. Three distinct regimes:

**LLM finetuning** — autoregressive next-token cross-entropy. Preference methods act on
sequence log-probs (sum of per-token log-probs).

**VLM finetuning (this project)** — still autoregressive text generation, just conditioned on
image embeddings injected into the language model's context (as prefix tokens or via
cross-attention). Structurally near-identical to LLM finetuning, which is exactly why TRL's
`DPOTrainer` works on Idefics2 and LLaVA with no changes to the loss. The differences are
practical, not mathematical:
- Image tokens dominate sequence length → aggressive resizing, gradient checkpointing, OOM management
- Vision tower is usually frozen; you align the language model and the projection layer
- The behavior you're correcting is **hallucination**, not tone or helpfulness

**Image generation (diffusion)** — genuinely different mathematics:
- **No tractable likelihood.** Diffusion isn't autoregressive, so `log π(y|x)` doesn't exist
  in usable form. **Diffusion-DPO** (Wallace et al.) substitutes the denoising MSE as an
  ELBO proxy for log-likelihood and slots it into the same DPO expression.
- **Reward applies to the final image, not per token.** No sequence axis for credit
  assignment. **DDPO** (Black et al.) reframes the denoising chain as a multi-step MDP and
  runs policy gradient over it.
- **You can backpropagate through the sampler.** The decoder is differentiable end-to-end, so
  DRaFT/AlignProp push a reward model's gradient directly back through sampling steps. There
  is no LLM analogue — you can't backprop through `argmax`.
- **Evaluation is much weaker.** No exact match or F1; you rely on CLIP score, PickScore,
  HPSv2, aesthetic predictors, or human raters.
- **Entirely different technique vocabulary** — DreamBooth, textual inversion, ControlNet,
  LoRA on cross-attention blocks.

**Exception worth mentioning:** token-based autoregressive image models (Chameleon, Emu3) *do*
have sequence log-probs, so DPO applies to them almost directly. The autoregressive/diffusion
split matters more than the text/image split.

---

## 8. Likely questions

**"Why implement the losses yourself instead of using TRL?"**
Learning objective first — I wanted to understand the mechanics rather than tune a black box.
But it was also necessary: TRL's trainers assume a decoder-only causal LM. BLIP is an
encoder-decoder with cross-attention to image embeddings, so the log-prob computation and the
forward pass had to be written against that architecture. When I scaled to Idefics2 and
LLaVA — which *are* decoder-only — I switched to TRL, because by then it was the right tool.

**"Why is there no reference model in ORPO, and why does that matter?"**
DPO needs a reference to define its implicit reward — the reward is literally the log-ratio
against it. ORPO instead uses the odds ratio *between* chosen and rejected within a single
forward pass, so nothing external anchors it; the SFT cross-entropy term does the anchoring
instead. Practically: half the memory and one training stage instead of two. For an 8B VLM
where model + reference + gradients + AdamW states already push past a single A100, that's
the difference between feasible and not.

**"When would you pick KTO over DPO?"**
Data availability, almost entirely. DPO requires paired comparisons on the same prompt; KTO
needs only binary good/bad labels. Production feedback is overwhelmingly thumbs-up/down, so
KTO often has 10–100× more usable data. KTO also handles class imbalance explicitly via
`λ_D`/`λ_U` weights (I set both to 1, since my data was paired and therefore balanced).

**"Your KTO implementation takes pairs — isn't that missing the point?"**
Yes, and I documented it in the class docstring. I implemented KTO's *loss* but fed it paired
data, so I got none of the data-efficiency benefit that's KTO's main selling point. Doing it
properly needs a separate unpaired dataset class with a per-example desirable/undesirable flag.
That's the first thing I'd build if I picked this back up.

**"What would you do differently?"**
Four things, in order: (1) start from a model that can actually represent the target responses
— LLaVA-1.5-7B rather than BLIP-base; (2) fix the ORPO length-normalization bug; (3) evaluate
on a hallucination-specific benchmark like POPE or MMHal-Bench instead of EM/F1, since that's
the behavior these methods target; (4) build the unpaired KTO path so the comparison is fair
to KTO.

---

## 9. Repo map

```
cs231nmilestone-main/       # the substantive work
├── customblip.py           # ORPOBlipForVQA, PairBlipForVQA, all three losses
├── customdataloaders.py    # VLFeedback + MLLM-DPO loaders, paired collation
├── trainPA.py              # unified harness: 4 methods, LoRA, W&B, EM/F1/LLM-judge eval
├── train-{dpo,kto,orpo,vanilla}.sh
└── train.ipynb             # 49-cell experiment log with preserved outputs

idefics/                    # TRL-based scaling attempt
├── dpo.py                  # Idefics2-8B + RLAIF-V, complete
├── cpo.py, kto.py          # variants
└── orpo.py, rloo.py        # stubs

llava/                      # LLaVA-1.5 DPO configs + memory-budget notes
```

## 10. References

- Rafailov et al. 2023 — [DPO](https://arxiv.org/abs/2305.18290)
- Ethayarajh et al. 2024 — [KTO](https://arxiv.org/abs/2402.01306)
- Hong et al. 2024 — [ORPO](https://arxiv.org/abs/2403.07691)
- Azar et al. 2023 — [IPO](https://arxiv.org/abs/2310.12036)
- Xu et al. 2024 — [CPO](https://arxiv.org/abs/2401.08417)
- Wallace et al. 2023 — [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
- Black et al. 2023 — [DDPO](https://arxiv.org/abs/2305.13301)
- [HF: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)
