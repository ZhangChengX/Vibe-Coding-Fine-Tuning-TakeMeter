# Vibe-Coding-Fine-Tuning-TakeMeter

TakeMeter is a fine-tuned text classifier that sorts r/LocalLLaMA posts and comments by
**discourse quality** — how well a post backs up what it claims. It is not about whether a
take is *correct*; it is about whether the post shows any *support* for its claim.

---

## 1. TakeMeter — Discourse Quality Classifier

- **One axis, three labels.** Every post is one point on a single line: *does it back up its
  claim?*

  ```
  ANALYSIS          OPINION              REACTION_NOISE
  backs a claim  →  asserts, no backing  →  makes no claim
  ```

  | Label | Plain definition |
  |---|---|
  | `ANALYSIS` | Makes a claim **and** supports it (a reason/mechanism, a number/spec, a source, or a firsthand result). |
  | `OPINION` | Makes a real claim about the topic but gives **no** support. |
  | `REACTION_NOISE` | Makes **no** real claim — jokes, memes, one-liners, pure emotion, bare questions. |

- **A labeled dataset** of r/LocalLLaMA-style posts (see §2).
- **A fine-tuned `distilbert-base-uncased`** classifier (see §3).
- **A zero-shot baseline** using Groq `llama-3.3-70b-versatile` (see §4).
- **An evaluation that compares two dataset sizes** (209 vs 447 examples) to show how much the
  amount of data mattered (see §5).

---

## 2. Dataset

**Where it comes from.** All examples are r/LocalLLaMA-style posts about open-weight models
(quants, hardware, tok/s, perplexity, benchmarks, model releases).

**How I labeled.** For each post I applied two steps in order:
1. Is there any real claim about the topic? **No → `REACTION_NOISE`.**
2. Is there concrete support (reason, number, source, or firsthand result)? **Yes →
   `ANALYSIS`. No → `OPINION`.**

**Two dataset sizes:**

| File | Examples | ANALYSIS | OPINION | REACTION_NOISE |
|---|---|---|---|---|
| `takemeter_data_209.csv` | 209 | 70 | 70 | 69 |
| `takemeter_data_447.csv` | 447 | 148 | 150 | 149 |

**Three examples that were hard to label** (full notes in planning.md §3):
1. *"Since Q4 preserves the outlier channels... why does my 70B still OOM at 24GB? Because the
   weights are ~20GB and the KV cache adds ~5GB."* → **ANALYSIS.** It looks like a question,
   but it carries a real, supported claim. Judge the claim, not the question mark.
2. *"Llama is open weight."* → **OPINION.** True and on-topic, but it argues no point and gives
   no support, so it is not ANALYSIS.
3. *"Everyone's benchmarks are cooked, just vibe-check it yourself."* → **OPINION.**
   "Vibe-check it" sounds like a reason but is just a vague appeal, so it does not count as
   support.

---

## 3. Fine-tuning pipeline

- **Starting model:** `distilbert-base-uncased` (a small, fast BERT-family model) with a fresh
  3-class classification head.
- **Training approach:** standard Hugging Face `Trainer`, 70/15/15 stratified train/val/test
  split, tokenized at max length 256, best model kept by validation accuracy.
- **Key hyperparameter decision:** I kept the standard settings — **learning rate `2e-5`, 3
  epochs, batch size 16** — on purpose. With only ~49 examples per class in the 209 set, more
  epochs or a higher learning rate would overfit fast. The most important "knob" turned out not
  to be a hyperparameter at all but the **dataset size** (209 → 447), which is what actually
  fixed the model (see §5).

Full run logs: [finetuning_209.ipynb](finetuning_209.ipynb) and
[finetuning_447.ipynb](finetuning_447.ipynb).

---

## 4. Baseline

The baseline is a **zero-shot** prompt to Groq `llama-3.3-70b-versatile`: it is given the label
definitions and one example per label, then asked to output just a label name — no training.
This tells us how hard the task is for a strong general model with no fine-tuning.

---

## 5. Evaluation report

### 5.1 Overall accuracy

| Dataset | Test size | Baseline (Groq zero-shot) | Fine-tuned DistilBERT |
|---|---|---|---|
| 209 examples | 32 | **100%** | **65.6%** |
| 447 examples | 68 | **100%** | **100%** |

Two clear findings:
- On the small (209) set, fine-tuning was **worse than doing nothing** — it lost 34 points to
  the zero-shot LLM.
- Doubling the data (447) raised the fine-tuned model from 65.6% to **100%**, matching the
  baseline.

### 5.2 Per-class metrics

**Fine-tuned model on the 209 set (the interesting, broken one):**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ANALYSIS | 0.56 | 1.00 | 0.71 | 10 |
| OPINION | 0.70 | 0.64 | 0.67 | 11 |
| REACTION_NOISE | 1.00 | 0.36 | 0.53 | 11 |
| **Accuracy** | | | **0.66** | 32 |
| **Macro avg** | 0.75 | 0.67 | 0.64 | 32 |

This is why accuracy alone is misleading: overall accuracy is 0.66, but **REACTION_NOISE recall
is only 0.36** — the model missed almost two-thirds of the noise posts. The per-class view
exposes a failure the single number hides.

**Fine-tuned model on the 447 set, and the baseline on both:** every class scored **1.00**
across precision, recall, and F1.

### 5.3 Confusion matrix — fine-tuned model (209 set)

Rows = true label, columns = what the model predicted.

| True ↓ / Pred → | ANALYSIS | OPINION | REACTION_NOISE |
|---|---|---|---|
| **ANALYSIS** | 10 | 0 | 0 |
| **OPINION** | 4 | 7 | 0 |
| **REACTION_NOISE** | 4 | 3 | 4 |

The pattern is striking: **every single error moves *up* the quality spectrum** (noise → opinion
→ analysis), never down. The model over-calls ANALYSIS (8 false positives, precision only 0.56)
and badly under-detects REACTION_NOISE. Image: [confusion_matrix_209.png](confusion_matrix_209.png).

**Fine-tuned model (447 set):** a perfect diagonal — 22 ANALYSIS, 23 OPINION, 23 REACTION_NOISE,
zero errors. Image: [confusion_matrix_447.png](confusion_matrix_447.png).

### 5.4 Three specific failures and why they happened (209 model)

All confidences below are real values from the run, and all sit near **0.33–0.40 — barely above
random for 3 classes** — meaning the model was basically guessing on these.

1. **"crying in 8GB VRAM"** — true `REACTION_NOISE`, predicted `ANALYSIS` (conf 0.35).
   This is a joke about not having enough memory. But it contains the hardware term "8GB VRAM,"
   and the model learned to treat technical words as a sign of substance. **The jargon fooled
   it.**

2. **"ggufs when??"** — true `REACTION_NOISE`, predicted `ANALYSIS` (conf 0.34).
   A one-line request with no claim at all. Again, the technical token "ggufs" pulled it toward
   ANALYSIS. The model never learned that a bare question is noise.

3. **"Honestly cloud APIs are still better for anything serious."** — true `OPINION`, predicted
   `ANALYSIS` (conf 0.36).
   This is a confident, full-sentence claim — but it offers **no** reason, number, or source, so
   it is OPINION. The model mistook a confident *sentence shape* for actual support.

**What this tells us:**
- **Which boundary is hard:** the errors cluster on two boundaries — REACTION_NOISE being read
  as ANALYSIS/OPINION, and OPINION being read as ANALYSIS. Both point the same way: the model
  promotes posts it shouldn't.
- **Why it's hard:** with ~49 training examples per class, the model didn't have enough signal
  to learn the real rule (*is there support?*). Instead it latched onto easy surface cues —
  technical vocabulary and confident sentence structure.
- **Labeling problem or data problem?** Not a labeling problem. The same labels, fed in larger
  numbers, gave a **perfect** 447 model. So the boundary is learnable; the 209 set was simply
  too small. It is a **data-quantity problem**.
- **What fixed it:** more examples — especially noise posts that contain jargon and opinions
  that sound confident — so the model learns that jargon ≠ analysis and confidence ≠ support.
  Going from 209 to 447 did exactly this and removed all errors.

### 5.5 Failure-pattern analysis (AI-assisted, then verified)

I pasted the 11 misclassified posts into an LLM and asked it to find common themes. It
suggested three patterns: (a) **errors always climb the spectrum** (nothing was ever
*demoted*), (b) **short posts with model/hardware words** get read as ANALYSIS, and (c)
**confident one-sentence opinions** get read as ANALYSIS. I then re-read all 11 examples myself
to check: (a) and (b) held up exactly (7 of 11 errors are noise→up, and most involve a jargon
token); (c) held for the OPINION→ANALYSIS cases. I **discarded** one suggested pattern —
"errors are caused by emojis" — because several emoji posts were classified correctly, so emojis
were not the driver. The verified takeaway is in §5.4.

### 5.6 Sample classifications (fine-tuned 209 model)

Predicted label and the model's real confidence. (The provided run logged confidence only for
misclassified posts, so the correct example below shows the verified prediction without a logged
score.)

| Post | True | Predicted | Confidence | Right? |
|---|---|---|---|---|
| "ggufs when??" | REACTION_NOISE | ANALYSIS | 0.34 | ✗ |
| "touch grass everyone (after you download this)" | REACTION_NOISE | ANALYSIS | 0.35 | ✗ |
| "4-bit is a scam, run it in fp16 or don't bother." | OPINION | ANALYSIS | 0.40 | ✗ |
| "Honestly cloud APIs are still better for anything serious." | OPINION | ANALYSIS | 0.36 | ✗ |
| "Switched llama.cpp → vLLM on my 3090 and Qwen2.5-7B went from 34 to 61 tok/s..." | ANALYSIS | ANALYSIS | (correct; not logged) | ✓ |

**Why the correct one is reasonable:** that last post names a specific before/after number
(34 → 61 tok/s) and a mechanism (paged attention) — that is exactly the concrete support the
ANALYSIS label is meant to capture, so predicting ANALYSIS is the right call for the right
reason.

### 5.7 What the model learned vs. what I intended

- **What I intended:** the model should grade *support* — does the post give a reason, a number,
  a source, or a firsthand result?
- **What the 209 model actually learned:** "technical words and confident sentences = ANALYSIS."
  It picked up the *surface* of substance, not the substance itself. That is why a joke like
  "crying in 8GB VRAM" got promoted.
- **The honest catch with the 447 model:** it scores 100% — but so does the zero-shot baseline.
  When *both* a tiny fine-tuned model and a general LLM get a perfect score, that is a warning
  sign that the **test set is too easy**, not that the model is perfect. My AI-authored data has
  clean, well-separated labels and very few truly borderline posts. Real r/LocalLLaMA data —
  full of sarcasm, vague "support," and mixed posts — would almost certainly drop both models
  below 100%. So even the 447 model may still be leaning on surface cues; the current test set
  just isn't hard enough to reveal it.

---

## 6. Spec reflection

- **One way the spec helped:** the planning.md rule to evaluate with **per-class metrics and a
  confusion matrix, not just accuracy**, is exactly what caught the 209 model's real problem.
  Accuracy said "66%, not great"; the per-class view said "REACTION_NOISE recall is 0.36 and all
  errors climb the spectrum," which is a precise, fixable diagnosis. The single-axis label
  design also kept annotation consistent enough that more data led straight to a perfect score.

---

## 7. AI usage

1. **Failure analysis.** I gave the LLM the 11 wrong predictions and asked for common patterns
   (§5.5). It proposed "errors climb the spectrum" and "jargon mistaken for analysis," which I
   confirmed by re-reading the examples — and I **discarded** its "emojis cause errors"
   suggestion after checking that emoji posts were often classified correctly.
2. **Drafting this report.** I used an AI tool to help organize the results into clear sections;
   all numbers come directly from the committed JSON files and notebooks, and I verified the
   confusion matrix by hand against the per-class precision and recall.

---

## 8. Files

| File | What it is |
|---|---|
| `takemeter_data_209.csv`, `takemeter_data_447.csv` | the two labeled datasets |
| `finetuning_209.ipynb`, `finetuning_447.ipynb` | full training + evaluation runs |
| `evaluation_results_209.json`, `evaluation_results_447.json` | saved metrics |
| `confusion_matrix_209.png`, `confusion_matrix_447.png` | confusion matrix images |
| `planning.md` | design notes, label rules, metric reasoning |
