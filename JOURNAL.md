# Diversity Maximization Journal

**Goal**: Maximize expression diversity — unique templates, varied operands, high edit distance — while maintaining correctness.

**Best so far (Exp4)**: template_ratio=0.60, 6 unique templates, avg_edit_dist=0.53, 2/10 correct

---

## Run 1: Baseline Re-establishment

**Time**: Starting now
**Plan**: Run Exp4 config but with 3 epochs instead of 1, and bigger eval (20 targets). The model was still learning at epoch 1 — closeness was 0.85 and rising. More training might push both correctness and diversity.

**Config**: correctness=2.0, closeness=0.5, format=0.3, complexity=2.0, nontrivial=1.0, no_target=0.5, 3 epochs, n_train=1500

**Status**: Complete

---

## Run 2: First Diversity Sweep (6 experiments)

**Time**: Running on A100-80GB
**Plan**: 6 rapid experiments (250 steps each, ~5 min), varying reward weights. 20 targets × 2 samples = 40 outputs per experiment.

### Results so far:

| Experiment | Template Ratio | Unique/40 | Edit Dist | Entropy | Correct | Leak |
|---|---|---|---|---|---|---|
| div01_baseline (2,0.5,0.3,2) | **0.400** | **16** | **0.410** | **1.869** | 2.5% | 5% |
| div02_high_close (2,2,0.3,2) | 0.300 | 12 | 0.251 | 0.986 | 35.0% | 17.5% |
| div03_notarget_3x (strong penalty) | 0.075 | 3 | 0.324 | 1.529 | 15.0% | 0% |
| div04_heavy_cmplx (1,1,0.3,4) | 0.025 | 1 | 0.128 | 1.585 | 100% | 100% |

### Analysis:

**div01 is still the best for diversity** — 16 unique templates, 0.40 ratio, highest entropy.

**div02** boosted correctness to 35% but at the cost of diversity (dropped to 12 templates). Higher closeness makes the model converge faster to formulaic solutions.

**div03** with 3x no-target penalty collapsed to just 3 templates. Too much penalty = less exploration.

**div04** is the worst — heavy complexity with low correctness weight caused it to find ONE template with 100% correctness and 100% target leak. It's just `(N-1)/1+1` again. The complexity weight dominated so hard that correctness pressure was too weak to prevent the hack, and without strong no-target penalty it leaked.

**Key insight**: There's a clear **diversity-correctness tradeoff**. The more you push correctness, the more the model converges to templates. The sweet spot is div01's config where correctness is moderate and the model is still exploring.

### Full Wave 1 Results:

| Experiment | Template Ratio | Unique/40 | Edit Dist | Entropy | Correct | Leak |
|---|---|---|---|---|---|---|
| **div01_baseline** (2,0.5,0.3,2) | **0.400** | **16** | **0.410** | **1.869** | 2.5% | 5% |
| div02_high_close (2,2,0.3,2) | 0.300 | 12 | 0.251 | 0.986 | 35.0% | 17.5% |
| div06_extreme_cmplx (0.5,0.5,0.3,5) | 0.250 | 10 | 0.381 | 1.548 | 7.5% | 0% |
| div03_notarget_3x (strong penalty) | 0.075 | 3 | 0.324 | 1.529 | 15.0% | 0% |
| div05_fast_lr (lr=2e-4) | 0.050 | 2 | 0.198 | 1.577 | 85.0% | 7.5% |
| div04_heavy_cmplx (1,1,0.3,4) | 0.025 | 1 | 0.128 | 1.585 | 100% | 100% |

### Key Findings:
1. **Clear diversity-correctness tradeoff**: correlation is nearly perfectly inverse
2. **div01 config is the diversity champion**: 16 unique templates with genuine arithmetic
3. **Higher closeness kills diversity**: div02 jumped to 35% correct but dropped to 12 templates
4. **Fast learning = template collapse**: div05 with lr=2e-4 converged to factorization template `a*b-1+1` (85% correct but only 2 templates)
5. **div05's factorizations are interesting**: `25→5*5-1+1`, `42→2*21-1+1`, `55→5*11-1+1` — actual factor discovery!

### Scoring: template_ratio × (1 + correctness)
This rewards both diversity AND correctness. A model with 0.4 diversity and 10% correct scores 0.44. One with 0.05 diversity and 85% correct scores only 0.09.

---

## Run 3: Wave 2 — Finding the Sweet Spot

**Plan**: Explore the closeness weight between 0.5 (div01, diverse) and 2.0 (div02, correct) to find the optimal tradeoff. Also try lower correctness weight, 2 epochs, and higher lr.

**Experiments**:
- w2_close_0.7, w2_close_1.0, w2_close_1.3 — closeness sweep
- w2_low_correct — correctness=1.0 instead of 2.0
- w2_2epoch_best — 2 epochs with div01 config
- w2_easy_only — only easy targets
- w2_explore_lr — lr=1e-4 with closeness=0.8

**Status**: Running on A100-80GB (~42 min total)...
