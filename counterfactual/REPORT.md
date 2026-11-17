# One counterfactual separates 2,401 reasoning programs

Two programs can compute the same answer while representing different rules. Consider a signed word problem whose intended program is `A − B + 2C`. If A and B are both 18 and C is 20, the wrong program `−A + B + 2C` also returns 40. One carefully chosen input, `[1, 7, 49, 343]`, gives 92 and 104 respectively. That same test distinguishes **every pair of four-term linear programs with integer coefficients between −3 and 3**.

Laurence Aitchison and colleagues' *Reasoning Arena* studies what to do when identical outcome rewards hide differences between reasoning traces. Here the output is a restricted executable program, and an independently known counterfactual answer distinguishes equal-score programs without a learned judge. A generated benchmark supplies that reference. The model translates text into a program and an interpreter executes it, following prior program-based prompting work. [Reasoning Arena](https://arxiv.org/abs/2606.09380) · [Program of Thoughts](https://openreview.net/forum?id=YfZ4ZPt8zd)

## Why one input suffices

For two different programs, consider their coefficient difference. Let $j$ be the largest index, counting from zero, where the two coefficients differ. The contribution from that position is at least $7^j$ in magnitude. The maximum sum of all smaller contributions is $6(1 + 7 + \cdots + 7^{j-1}) = 7^j - 1$. They cannot cancel. An answer on the radix probe therefore certifies full agreement with any reference program in the class. Four unit-vector evaluations give the same decision, using four calls to the integer evaluator.

The probe also minimises the sum of the magnitudes of integer inputs needed for one-check identification. For any integer input $x$, all output values lie between $-3\lVert x\rVert_1$ and $3\lVert x\rVert_1$. There can be at most $6\lVert x\rVert_1+1$ distinct results, and the 2,401 programs require $\lVert x\rVert_1\geq400$. The chosen input attains 400. Exhaustive verification obtained 2,401 unique results. Across 32 random single probes drawn from the same positive integer range 1 to 343, the mean was 1,029.5 programs with unique results, with a range of 98 to 1,929.

The coefficient bound matters. For example, `4A` and `−3A+B` both give 4 on this probe. A correct probe answer comes from a trusted specification. When a reference coefficient vector is already available, direct vector comparison is simpler. The one-check result matters when the reference can be queried on chosen inputs but its coefficients are unavailable.

## Fresh Qwen benchmark

We generated 64 signed-count problems across four families, with 32 ordinary cases containing distinct values and 32 cases containing a deliberately repeated value. The latter stress-test cases allow an error in source attribution to leave the computed answer unchanged. The gold rule comes from the question generator. The model receives neither the rule nor the counterfactual answer. Four inference policies share the same local Qwen3-1.7B 4-bit checkpoint. They produce a native worked answer, a structured program, a structured program with the four magnitudes hidden from the translator, or a second structured program after reviewing the first attempt. Every structured policy gets the same generic worked example, uses the same bounded grammar and calls the same integer executor. No test family has its own example. A separate case warms the model and does not enter the results. The review baseline pays for both passes.

The original frozen prompt yielded zero parseable programs under all three structured policies, and only two parseable native answers. All calls exhausted their budgets. The diagnostic reproduced the model's long prose starts under both single and batched generation. We kept that result and froze a separate follow-up on the same items. The follow-up supplies an identical assistant-side output prefix to the three structured policies. It also adds a short formatting reminder and raises the native cap. [Original protocol](PROTOCOL.md) · [Original outcomes](results.json) · [Follow-up protocol](RECOVERY.md)

| Format follow-up policy | Parsed / 64 | Correct numeric answer / 64 | Correct program / 64 | Copied worked example / 64 | Input + generated tokens |
|---|---:|---:|---:|---:|---:|
| Native worked answer | 23 | 3 | — | — | 17,189 |
| Structured program | 57 | 0 | 0 | 53 | 16,740 |
| Hide source magnitudes, structured program | 61 | 0 | 0 | 55 | 16,317 |
| Structured review, two calls | 64 | 0 | 0 | 59 | 35,118 |

The output prefix recovered 182 executable programs across the three arms, but none encoded the expected rule. The radix check, four basis checks and direct coefficient tests agreed on all 182. Neither hiding the numerical values nor paying for a second review fixed the grounding error. The native baseline also struggled to finish or meet its format. The four-policy comparison is specific to this checkpoint and prompt family. The 32 deliberately swapped source-pair programs all passed the original answer check. Both the radix and the independently sampled random check rejected all 32 of these simple mutations. The exhaustive collision check establishes the difference between the two kinds of probe. The Qwen processes used up to 2.07 GB peak active MLX allocation. [Follow-up outcomes](recovery_results.json)

## Changing only the worked example

We froze a paired intervention after seeing the copying pattern. The 32 new questions used a different generator seed and shared no question text with the previous 64. Each question received the original example, its sign reversal, and a no-example instruction. Both examples were legitimate for their own demonstration and wrong for every target question. The example replacement left the requested response format and model unchanged. The two demonstration arms processed exactly 7,445 input tokens each. [Intervention protocol](ATTRIBUTION.md)

| Prompt on the same 32 new questions | Parseable | Copied original example | Copied reversed example | Grounded |
|---|---:|---:|---:|---:|
| Original example | 28 | **28** | 0 | 0 |
| Sign-reversed example | 17 | **0** | 3 | 0 |
| No example | 3 | **0** | 0 | 0 |

Changing the example cut exact copying of the original wrong program from 28/32 to 0/32, a paired difference of **87.5 percentage points** (95% stratified interval **78.1 to 96.9 points**, exact paired sign test **p = 7.45 × 10⁻⁹**). The reversed example copied its own program only 3/32 times, so the joint requirement that both examples trigger substantial copying failed. Replacing the example removed this particular failure but recovered zero correct programs. No-example output compliance was also weak. These comparisons isolate a specific exemplar effect within one checkpoint, response prefix and narrow question generator. They do not show which aspect of the exemplar caused the difference or recover the model's internal computation. The intervention used 1.89 GB peak active MLX allocation. [Intervention outcomes](attribution_results.json)

## Reproduce

From the repository root, use Python 3.11 or later. The offline replay checks the finite certificate, recreates the paired cohorts by seed and verifies results from parsed outputs and token counts. The traces contain no generated prose or model weights.

```sh
python3 counterfactual/experiment.py demo
python3 counterfactual/experiment.py analyse counterfactual/qwen_trace.json --verify counterfactual/results.json
python3 counterfactual/experiment.py analyse counterfactual/qwen_recovery_trace.json --verify counterfactual/recovery_results.json
python3 counterfactual/attribution.py analyse counterfactual/attribution_trace.json --verify counterfactual/attribution_results.json
```

To generate fresh traces on an Apple Silicon machine, point `QWEN_DIR` at the complete local MLX Qwen3-1.7B-4bit checkpoint. We used MLX 0.32.3 and MLX-LM 0.31.3. All three commands run one checkpoint each. Fresh elapsed times vary.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 counterfactual/experiment.py score "$QWEN_DIR" /tmp/original-qwen-trace.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 counterfactual/experiment.py score "$QWEN_DIR" /tmp/format-qwen-trace.json --recovery
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 counterfactual/attribution.py score "$QWEN_DIR" /tmp/example-qwen-trace.json
```
