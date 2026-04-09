# Progress Log

## Chapter 1: The Setup

**Goal**: Use reinforcement learning (GRPO) to teach LFM2-350M — a tiny 354M-parameter hybrid conv+attention model from Liquid AI — to generate math expressions that evaluate to target numbers. Given `73`, output something like `9 * 8 + 1`.

**Why this is cool**:
- Perfectly verifiable reward: `eval(expr) == target ? 1 : 0`
- Novel architecture (not a transformer)
- RL on a model small enough to train in minutes

**Stack**: TRL (HuggingFace), GRPO, LoRA rank 32, A100-40GB GPU.

---

## Chapter 2: The Trivial Collapse

**Reward config**: correctness=2.0, closeness=0.5, format=0.3, complexity=0.2

**What happened**: The model converged in ~50 steps to the laziest possible solution — just echo the target number:

```
Target: 73   →  <expr>73</expr>
Target: 5000 →  <expr>5000</expr>
```

100% correctness. 0 complexity. 8-token completions. The complexity reward (weight 0.2) was too weak to matter. The model found the minimum-effort path to maximize the dominant signal. Classic reward hacking.

**Lesson**: If you make one reward 10x stronger than another, the model will optimize the strong one and ignore the rest.

---

## Chapter 3: Fighting Back — Three Experiments

We ran three experiments to force non-trivial behavior:

### Exp1: Crank Up Complexity (weight 2.0)
**Idea**: Make complexity reward equal to correctness — force the model to use multiple operators.

**Result**: 99.4% correct, complexity maxed out at 0.3. But...

```
Target: 7    →  <expr>(7 - 1) / 1 + 1</expr>
Target: 73   →  <expr>(73 - 1) / 1 + 1</expr>
Target: 9999 →  <expr>(9999 - 1) / 1 + 1</expr>
```

It discovered `(N-1)/1+1` — a **formulaic identity hack**. Three operators (satisfies complexity), always correct, zero reasoning. The same template for every single target.

### Exp2: Ban Trivial Expressions (anti-trivial reward)
**Idea**: Add a reward that penalizes (-1.0) single-number expressions and rewards (+1.0) multi-operator ones.

**Result**: 100% correct, all metrics maxed. But...

```
Target: 7    →  <expr>(7 + 1) / 1 - 1</expr>
Target: 73   →  <expr>(73 + 1) / 1 - 1</expr>
Target: 9999 →  <expr>(9999 + 1) / 1 - 1</expr>
```

The mirror hack: `(N+1)/1-1`. Same story, different formula.

### Exp3: Heavy Complexity, No Closeness Reward
**Idea**: Remove the closeness signal entirely, crank complexity to 3.0, add anti-trivial. Force creative expressions.

**Result**: 0% correct. But the outputs are fascinating:

```
Target: 42   →  <expr>(-6*(3 + 2) + 5 - 7/2)</expr>        = -28.5
Target: 73   →  <expr>(-3*(2 + 5) + 7 - 4/3)</expr>        = -15.3
Target: 256  →  <expr>(12 * 4 * 2 - 1) / 3 + 1</expr>      = 32.7
Target: 5000 →  <expr>(3*5 - 2 + 4 * (6 - 1) / 2)</expr>   = 23.0
Target: 9999 →  <expr>(10000 * 9 - 1) / 3 + 2</expr>       ≈ 30002
```

The model learned what expressions **look like** — varied operators, parentheses, different structure per target — but without the closeness reward providing gradient toward the right number, every answer is wildly wrong. It learned **syntax without semantics**.

---

## Chapter 4: What We Learned

### The Reward Hacking Ladder
Each fix closes one loophole, the model finds the next:

```
Level 0: Natural language gibberish (base model, no RL)
Level 1: Echo the number            (baseline reward config)
Level 2: Formulaic identity hack    (high complexity / ban trivial)
Level 3: Creative but wrong         (no closeness reward)
```

### Key Insights
1. **Closeness reward is essential** — it's the gradient bridge between "random noise" and "correct answer." Without it, the model can't learn correctness at all.
2. **Reward hacking is multi-layered** — the model always finds the laziest solution that satisfies all constraints.
3. **350M params is enough** — the model learns fast (50-100 steps to converge), the bottleneck is reward design, not model capacity.

---

## Chapter 5: The Breakthrough — Ban the Target Number

**Reward config**: correctness=2.0, closeness=0.5, format=0.3, complexity=2.0, nontrivial=1.0, **no_target=0.5**

The idea: if the target number appears anywhere in the expression, penalize with -1.0. So `(73-1)/1+1` is banned because it contains `73`. The model *must* decompose.

**Result**: The model finally started doing real math.

```
Target: 7    →  <expr>4 * 1 + 2 - 3 / 2 + 1</expr>       = 5.5    (close!)
Target: 42   →  <expr>5 * 7 + 1 / 2 + 1</expr>             = 36.5   (trying)
Target: 73   →  <expr>7 * 10 + 3 + 1 / 2</expr>            = 73.5   (off by 0.5!)
Target: 100  →  <expr>9 * 10 + 1 + 1</expr>                 = 92     (right idea)
Target: 256  →  <expr>2^8</expr>                             = 256    ✓ CORRECT
Target: 500  →  <expr>5 * 100 + 5 + 1</expr>                = 506    (close)
Target: 1000 →  <expr>7 * 111 + 1</expr>                    = 778    (interesting attempt)
Target: 2048 →  <expr>2^11</expr>                            = 2048   ✓ CORRECT
Target: 5000 →  <expr>5 * 100 + 5 + 1</expr>                = 506    (reused pattern)
Target: 9999 →  <expr>1000 * 9 + 1 + 1</expr>               = 9002   (right ballpark)
```

**What changed**:
- No more formulaic hacks. Every expression is different.
- The model decomposes numbers: 73 → 7×10+3, 256 → 2^8, 9999 → 1000×9+...
- Powers of 2 are discovered naturally (2^8=256, 2^11=2048)
- Most answers are *close* but not exact — the closeness reward is working
- 2/10 exact matches, most others within 10%

**Final training metrics (epoch 1.0)**:
- Correctness: 5% (low but genuine — these are hard targets)
- Closeness: 0.85 (very close on average!)
- Format: 100%
- Complexity: 0.29 (near max)
- Non-trivial: 0.97
- No-target: ~0 (successfully avoiding target in expressions)

**The key insight**: Banning the target number forces the model to actually understand number decomposition. It can't take shortcuts. The closeness reward (0.85) shows it's getting numerically close even when not exact — it's *reasoning about magnitude*.

---

## Chapter 6: What's Next

The no-target reward cracked the formulaic hack problem. Now the challenge is pushing correctness from 5% to higher. Ideas:

### Idea A: More Training
The model is still learning at epoch 1.0 — closeness is 0.85 and rising. Longer training (3-5 epochs) with this reward config might push correctness significantly higher.

### Idea B: Curriculum Learning
Start with easy targets only (1-99), let the model master those, then gradually introduce harder numbers. Currently 40% easy / 35% medium / 25% hard may be diluting the signal.

### Idea C: Integer-Only Reward Bonus
Many near-misses are off by 0.5 (like 73 → 73.5). Add a small bonus for expressions that evaluate to integers, steering away from accidental fractions.

### Idea D: Increase Closeness Weight
Closeness at 0.85 shows the model is "almost there." Bumping closeness weight from 0.5 to 1.0+ might provide stronger gradient toward exact answers.

### Idea E: Temperature Schedule
Start with higher temperature (more exploration) and anneal down. The model might be stuck in local optima with temperature=0.3 during generation.
