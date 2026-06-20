# TakeMeter — planning.md


## 1. Community

**Choice:** r/LocalLLaMA (all examples collected from this subreddit).

r/LocalLLaMA is a high-volume subreddit where people run, quantize, fine-tune, and benchmark
open-weight language models on their own hardware. Its discourse is unusually varied in
quality: the same thread mixes carefully measured benchmark reports, confident predictions
with nothing behind them, and pure meme reactions.

**Why it's a good fit for a classification task.** The variance is the point. Because the
community is technical, many posts *can* be supported with a number, a mechanism, or a
firsthand result — and many posts conspicuously aren't. That gives a clean, naturally
occurring spread across "backs its claim → asserts without backing → makes no claim," which
is exactly the axis we want to classify. A community that was uniformly high-effort (a
moderated journal) or uniformly low-effort (a meme sub) would collapse the labels; here all
three show up in volume, often in the same comment tree.

---

## 2. Labels

**The one axis:** every label is a point on a single spectrum — *does the post back up its
claim about the topic?* Holding to one axis (support, **not** correctness, smartness, or hype)
is what keeps the labels mutually exclusive.

```
ANALYSIS          OPINION              REACTION_NOISE
backs a claim  →  asserts, no backing  →  makes no claim
```

| Label | One-sentence definition |
|---|---|
| `ANALYSIS` | A post that makes a claim about the topic **and** supports it with at least one of: reasoning/mechanism, specific numbers or specs, a source/citation, or firsthand experience with a result. |
| `OPINION` | A post that makes a real claim about the topic but offers **no** support — a bare assertion, prediction, or reaction-with-a-view. |
| `REACTION_NOISE` | A post that makes **no substantive claim about the subject**: jokes, memes, one-liners, pure emotion, meta-complaints, off-topic chatter. |

**ANALYSIS — examples**
- "Q4 holds up because the outlier weights are preserved in higher precision." *(mechanism)*
- "On my 3090 it went from 18 to 42 tok/s after the update." *(firsthand + numbers)*

**OPINION — examples**
- "Open models are obviously going to win long term."
- "This release is overrated."

**REACTION_NOISE — examples**
- "lol we're so cooked 💀"
- "first"

**Decision procedure (apply in order, stop at first fit):**
1. Does the post make any defensible claim about the topic? **No → `REACTION_NOISE`.** Yes → continue.
2. Is there at least one piece of concrete support (reason/mechanism, number/spec, source, or firsthand result)? **Yes → `ANALYSIS`. No → `OPINION`.**

**Why these distinctions matter.** Sorting takes by *support* — rather than by whether they're
correct or popular — is what would make a community tool genuinely useful: it surfaces the
reasoning that makes r/LocalLLaMA worth reading and separates it from confident-but-empty
assertions and noise, without the classifier having to adjudicate who is *right*.

**Mutual exclusivity.** The single-axis design means each post lands at exactly one point on
the spectrum. The only two real boundaries are ANALYSIS vs. OPINION (is there concrete
support?) and OPINION vs. REACTION_NOISE (is there a topical claim at all?).

---

## 3. Hard edge cases

These are the post types I expect to be genuinely ambiguous between two labels, with the
handling rule I'll apply consistently during annotation.

**Three specific examples that gave genuine pause during annotation (Milestone 3):**

1. *"Since Q4 preserves the outlier channels in higher precision, why does my 70B still OOM at
   24GB? Because the weights are ~20GB and the 16k KV cache adds ~5GB..."* — candidate labels
   `REACTION_NOISE` (it's phrased as a question) vs. `ANALYSIS`. **Decided `ANALYSIS`:** the
   question embeds a real, supported claim with specific numbers; the question mark doesn't
   demote it. Judge the claim, not the punctuation.
2. *"Llama is open weight."* — candidate labels `OPINION` vs. `REACTION_NOISE`. **Decided
   `OPINION`:** it's a true, on-topic assertion, so it clears step 1 (a topical claim exists),
   but no claim is being *argued* and no support is offered, so it can't be ANALYSIS. We treat
   correct-but-bare facts as OPINION and apply that consistently.
3. *"Everyone's benchmarks are cooked, just vibe-check it yourself."* — candidate labels
   `ANALYSIS` vs. `OPINION`. **Decided `OPINION`:** "vibe-check it yourself" looks like a
   reason but is a gestural appeal, not a checkable mechanism/number/source, so it does not tip
   into ANALYSIS. Gestural support stays OPINION.

The post types below are the broader patterns these examples come from.

**Gestural / vague support — ANALYSIS vs. OPINION.** A post that *sounds* like it's reasoning
but only offers a vague appeal ("feels snappier," "everyone running local knows this").
→ **OPINION.** Support must be checkable or specific; a vague impression or appeal to
consensus does not tip it to ANALYSIS.

**Correct-but-bare facts — OPINION vs. REACTION_NOISE.** A true, on-topic statement with no
argument being made ("Llama is open weight"). → **OPINION.** It's a topical assertion, so it
clears step 1, but ANALYSIS requires support *for a claim being argued*, not just a true
sentence.

**Bare questions — OPINION vs. REACTION_NOISE.** A pure question asserts no claim
("anyone else getting OOM on 70B?"). → **REACTION_NOISE.** A question that *embeds* a supported
claim can climb to ANALYSIS; judge the claim, not the question mark.

**Joke + claim hybrids — OPINION vs. REACTION_NOISE.** "GPT-5 is gonna flop hard 😂". If a
real topical claim survives stripping the joke ("GPT-5 will underperform") → **OPINION**;
if nothing about the subject remains → **REACTION_NOISE.** Tone/emoji never demote a claim.

**Mixed-support posts — ANALYSIS vs. OPINION.** Label by the **central, load-bearing claim**.
If the main point is supported → ANALYSIS; if the support is incidental to an otherwise bare
assertion → OPINION.

**Long-but-empty — OPINION.** Length is not support. A five-paragraph rant with no
reason/number/source stays OPINION.

**Off-topic but reasoned — REACTION_NOISE.** If it's not about the community's subject, it's
noise/meta regardless of how reasoned it is. Stay on the topic axis.

**Handling rule during annotation:** when a post genuinely sits on a boundary, apply the
decision procedure top-down and record the post, the two candidate labels, and the
deciding factor in the CSV `notes` column. Borderline calls get logged, not skipped.

---

## 4. Data collection plan

**Source.** Public posts and top-level comments from r/LocalLLaMA only (no authenticated or
private content).

**Target volume & balance.** 200 is the floor; the working target is **~150 per label
(≈450 total)** in a single unsplit CSV (`text`, `label`, `notes`). The larger size isn't for
training convergence — DistilBERT learns from a few hundred — but so the 15% test split holds
~22 examples per class, where per-class precision/recall/F1 are statistically meaningful (at
200 total the test set is only ~10/class and a single error swings a metric ~10 points). Hard
constraint: **no single label above 70%** of the dataset.

**Expected skew & mitigation.** I expect REACTION_NOISE to be the easiest to over-collect and
ANALYSIS the rarest (genuinely supported posts are less common). If a label is
underrepresented after a first pass, I'll do **targeted collection** rather than random
scrolling:
- For ANALYSIS: benchmark threads, "I tested X" posts, quantization/perplexity discussions.
- For REACTION_NOISE: release-day megathreads and meme posts.
- For OPINION: "is X overrated?" / prediction threads.

---

## 5. Evaluation metrics

Accuracy alone is insufficient because the dataset is likely imbalanced and the classes are
not equally valuable. The full metric set:

- **Overall accuracy** — headline number, reported for both models, but not the primary
  signal.
- **Macro-averaged F1** — **primary metric.** It weights each class equally, so a model can't
  win by predicting the majority class (e.g. REACTION_NOISE). This is the number the
  fine-tuned model must beat the baseline on.
- **Per-class precision / recall / F1**, with particular attention to **ANALYSIS recall** —
  ANALYSIS is the rarest and most valuable class (it's what a "surface the good takes" tool
  exists to find), so missing it is the costliest error.
- **Confusion matrix (3×3)** — tells us *which* boundary the model confuses. The interesting
  failure is ANALYSIS↔OPINION (the support boundary); ANALYSIS↔REACTION_NOISE confusion would
  signal the model isn't reading content at all.

**Why these for this task:** the product goal is "promote supported takes, demote noise," so
the costs are asymmetric and per-class behavior matters more than a single pooled accuracy.

---

## 6. Definition of success

Concrete, objectively checkable thresholds on the locked test set:

1. **Beats the baseline:** fine-tuned model's **macro-F1 > the zero-shot llama-3.3-70b
   baseline's macro-F1.** This is the minimum bar — if it doesn't beat zero-shot, fine-tuning
   added nothing.
2. **Absolute quality:** **macro-F1 ≥ 0.70**, with **no single class's F1 below 0.55** (so it
   isn't carried by one easy class).
3. **Useful on the valuable class:** **ANALYSIS recall ≥ 0.65** — it must actually catch most
   supported takes to be worth deploying as a surfacing tool.

"Good enough for deployment" in a real community tool means hitting (1)–(3) **and** an
ANALYSIS precision high enough that surfaced posts are mostly genuinely supported (target
≥ 0.60), so users trust the "good take" signal. If precision is too low, the tool cries wolf;
if ANALYSIS recall is too low, it's not surfacing enough to be worth running.

---

## AI Tool Plan

**Label stress-testing** Before annotating examples, give Claude Opus the
label definitions and edge-case list and ask it to generate 5–10 posts that sit on the
ANALYSIS/OPINION and OPINION/REACTION_NOISE boundaries. Any post I can't classify cleanly means
a definition needs tightening.

**Annotation assistance** Use Claude to pre-label batches of
unlabeled posts given the definitions from this file. Every pre-assigned label is reviewed and
corrected by experts — pre-labeling speeds throughput, it does not replace reading. I'll
track which rows were pre-labeled (a `prelabeled` flag/note in the CSV) for disclosure in the
AI usage section. Skimming would produce noisy training data, so review is non-negotiable.

**Failure analysis** After evaluation, hand the list of wrong predictions
(text, true label, predicted label) to an AI tool and ask it to cluster the errors into
patterns (e.g. "vague-support posts misread as ANALYSIS," "ANALYSIS↔OPINION confusion
dominates"). I'll then **verify each proposed pattern against the actual examples myself**
before it goes in the evaluation write-up — the AI proposes patterns, I confirm them against
the data.

---
