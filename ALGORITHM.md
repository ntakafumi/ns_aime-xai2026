# Overview of the NS-AIME Algorithm

Reference Paper: T. Nakanishi, **"NS-AIME: Rule-Guided Re-Constraint of Approximate Inverse Explanations"**, 4th World Conference on eXplainable AI (Late-breaking, Demos and Doctoral Consortium), Fortaleza, Brazil, 2026.

This document summarizes the "structure of the paper’s algorithm," "**how the paper differs** from AIME, SHAP, LIME, and rule-based learners," and the "**correlation between theory and implementation**" (as well as modifications made to the existing implementation in this repository). Ambiguous points are explicitly marked as **assumptions**.

---

## 1. Positioning (In a Nutshell)

NS-AIME is neither a new predictor nor a new rule learner. It is a **post-hoc layer** that explains a **trained black box** `f` using the inverse operator of AIME, and then **re-constrains only the “shape” of its global attribute matrix via convex optimization to align it with logical rules derived from the data**.

> It inserts an **intermediate layer** between the “mathematically well-defined but logically unstructured explanations” produced by AIME (Inverse Operator-based XAI)—which lack the monotonicity and thresholds humans expect—and “rule-based explanations that are easy for humans to audit,” without retraining the black box.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `X ∈ R^{n×d}` | Data matrix (`x ∈ R^d`, tabular or vectorized inputs such as TF–IDF) |
| `Ŷ = f(X) ∈ R^{n×m}` | **Predicted class probabilities** of the trained classifier |
| `A† ∈ R^{d×m}` | AIME inverse operator (global attribute matrix). The `k`th column is the global contribution vector for class `k` |
| `M_{jk} ∈ {−1,0,+1}` | Monotonicity sign for feature `j` and class `k` |
| `t_{jk}` | Threshold for feature `j` and class `k` |
| `B ∈ R^{d×m}` | Post-reconstraint explanation matrix (output of NS-AIME) |

---

## 3. Algorithm (4 Steps)

### Step 1 — Inverse Operator A† (Eq. 1 in the paper)

```
A† = Xᵀ Ŷ (Ŷᵀ Ŷ + ε I_m)⁻¹            (ε > 0 is a ridge term for numerical stability)
```

- This is a **global linear inverse surrogate**, not a strict inverse of the black box.
- It does not assume that “the original model is globally linear”; rather, only the **inverse surrogate is linear** (as explicitly stated in §2 of the paper).
- Since it does not use perturbation sampling, unlike SHAP/LIME, it is **deterministic**, and re-running the process yields the same `A†`.

### Step 2 — Rule Induction (Paper §3.2)

From `A†`, `X`, and `Ŷ`, we **data-driven**ly induce three types of simple rules for “sufficiently important features” (the top `top_k` by `|A†_{·,k}|`) for each class.

1. **Monotonicity `M_{jk}`**: `+1` = contribution should be non-negative, `−1` = non-positive, `0` = no constraint. Assign signs only to important features.
2. **Threshold `t_{jk}`**: Estimated from the feature values of instances predicted to belong to class `k`.
   - Continuous features: A quantile (default is the median).
   - Binary features: The midpoint between the two values.
   - **Sparse features (e.g., TF–IDF): Presence rule `t = 0` (i.e., presence/absence)**.
3. **Pairwise co-occurrence (AND)**: For pairs of high-contribution features
   ```
   (x_{j1} ≥ t_{j1k}) ∧ (x_{j2} ≥ t_{j2k}) ⇒ k
   ```
   (If the corresponding monotonicity is negative, reverse the direction). **To ensure solvability and readability, we limit the analysis to pairwise interactions**; higher-order conjunctions and XOR-like interactions are excluded (as explicitly stated in §2, point 3, and §5 of the paper).

### Step 3 — Reconstraint with Rule Induction (Eq. 2 in the paper)

```
min_B  ‖B − A†‖²_F  +  α·R_mono(B; M)  +  β·R_rule(B; T, P)
```

- `R_mono`: A hinge penalty to suppress sign violations.
- `R_rule`: A hinge penalty for violated thresholds and the antecedent of the violated pairwise constraint.
- Since it consists of the first term (a quadratic proximity term) plus a convex hinge penalty, the **overall function is convex**. It can be solved stably using gradient descent.
- `B` starts from `A†`: The proximity term keeps `B` close to the raw inverse surrogate, while the hinge term pushes the class score toward the rule-compliant side in cases where the antecedent is satisfied.

### Step 4 — Dual Reporting and Audit Safeguards (Paper §3.4)

Do not treat the post-constraint explanation as a **silent replacement for the raw AIME**. Always **list** the following together:
(1) the raw `A†`, (2) the post-constraint `B`, and (3) a flag indicating significant divergence between the two.

```
RD(B, A†) = ‖B − A†‖_F / ‖A†‖_F                      (Eq. 3) The smaller the value, the closer to the raw AIME
GA(B, A†) = (1/m) Σ_k cos( B_{·,k}, A†_{·,k} )        (Eq. 4) The larger the value, the better the directional alignment
RS                                                     Rule Satisfaction Rate (higher is better)
```

- **`RD` is a distance**, and **a smaller value is better**. The paper initially mislabeled it as “fidelity” in the draft but renamed it **Relative Deviation (RD)**. `RD` does not cap at 1 (values exceeding 1 indicate a deviation greater than the Frobenius norm of the original matrix).
- When `RD` is high or `GA` is low (including negative values), treat this as an **audit flag**. This is because rule alignment may over-optimize the inverse surrogate, potentially normalizing problematic behavior (fairwashing) (Paper §3.4, Ref. [16]).

### Pseudocode

```
A† = Xᵀ Ŷ (ŶᵀŶ + εI)⁻¹                       # Eq.1
rules = induce_rules(A†, X, argmax(Ŷ))         # §3.2 (A† is read-only/invariant)
B = A†.copy()
repeat steps:
    g = 2(B − A†)                              # Proximal term
    g += α · subgrad R_mono(B; M)              # Sign hinge
    g += β · subgrad R_rule(B; T, P)           # Threshold and AND hinges
    B = B − lr · g                             # Convex ⇒ Convergence
report(A†, B, RD, GA, RS)                       # §3.4 Combined output + divergence flag
```

---

## 4. "Differences / Novelty" of the Paper

| Comparison Target | Differences from NS-AIME |
|---|---|
| **LIME / SHAP** | Does not rely on perturbation sampling; starts from the **deterministic inverse operator** of AIME (not affected by sampling, background, or locality choices). NS-AIME additionally explicitly imposes a **global logical structure** (monotonicity, threshold, AND). |
| **AIME (Base Method [3])** | AIME’s `A†` is mathematically well-defined but does not guarantee logical structure. NS-AIME enhances the **interpretability** of `A†` through convex optimization while preserving proximity. **The core AIME algorithm remains unchanged**. |
| **Rule Learners (SBRL/IDS/CORELS/QCBA, etc.)** | These optimize a **set of prediction rules**—that is, they replace the black box with a rule predictor. NS-AIME does not replace the black box; instead, it **optimizes the "form" of the explanation matrix produced by another model**. |
| **AMIE [13]** | Although the names are similar, they are different. AMIE mines Horn rules from a knowledge base. NS-AIME post-hoc re-constrains the **feature-attribute matrix** of a supervised classifier. |
| **Conventional "explanation correctness" metrics** | We abandon the misnomer "fidelity" and separate it into **distance = RD (smaller is better)** and **directional agreement = GA**. Furthermore, we place **dual reporting**—which presents significant discrepancies as audit signals rather than hiding them—at the core of our methodology. |

**What the paper does not claim** (Scope): We do not claim to solve general explanation fidelity or to outperform dedicated rule-based classifiers. The evidence is limited to 2 datasets (Titanic, 20 Newsgroups) × 3 classical models (logistic regression, random forest, LightGBM). Dense images and time series are excluded (since the dominant term `Xᵀ Ŷ` in Eq. 1 relies on sparse multiplication, making it practical for sparse inputs). Representative values from Table 1 of the paper: On 20NG, RS improves significantly from 0.02–0.15 to 0.75–0.93; on Titanic, while RS improves slightly, GA may become negative (= audit flag).

---

## 5. Correspondence Between Theory and Implementation (This Repository `ns_aime_xai2026`)

| Paper | Implementation |
|---|---|
| Eq. 1 (Inverse Operator) | `core.NSAIME._inverse_operator` |
| §3.2 (Rule Induction) | `utils.induce_rules` |
| Eq. 2 (Re-constraint) | `core.NSAIME.optimize` |
| Eq. 3 RD / Eq. 4 GA / RS | `utils.compute_metrics` (formerly `calculate_metrics`, backward compatible with tuples) |
| §3.4 dual reporting | `visualization.plot_dual_report`, `interactive.render_logic_graph_html` |
| §4.3 logic graph | `visualization.plot_logic_graph` (static, catchy), `interactive` (dynamic) |

### Explicit Assumptions (Due to Lack of Formula Details in the Paper)

1. **Normalization**: When `normalize=True`, `A†`, the score `X·B`, and the threshold (for numerical computation) are all consistently handled in the **normalized space**. However, the threshold displayed on the graph is shown in the **original units** (`threshold_display`). Since the paper does not explicitly state whether normalization is used, this is an assumption of this implementation.
2. **Interpretation of `R_rule` / RS**: We interpret the paper’s statement “the rule holds when the antecedent is satisfied” at the **score level**. Specifically, for instances satisfying the antecedent `x_j ≥ t`, we evaluate whether the class score `x·B_{·,k}` has a sign consistent with the rule (i.e., `≥ +γ` for positive rules and `≤ −γ` for negative rules). `R_mono` operates at the **coefficient level** (`sign(B_{jk}) = M_{jk}`).
3. **Margin `γ`**: A small margin used for satisfaction testing. The RS metric itself is measured by sign (`γ=0`).

### Changes from the Existing Implementation (= Corrections to Conform to the Paper)

Since the `ns_aime_xai2026` prior to this task contained **implementations that deviated from the paper**, we made corrections with user approval:

1. **Removed the in-place rewriting of `A†`**. The old `utils.induce_rules` overrode `A_dagger[idx,k]` under the name "Force A_dagger consistency." This breaks the dual reporting in §3.4 (since `A†` is no longer the raw inverse surrogate, rendering RD/GA meaningless). After the fix, `A†` is **completely invariant** (tests verify that `A†` matches the recalculation of `Eq.1` even after `fit`/`optimize`).
2. Renamed **`fidelity` → `RD`** (to follow the correction in Eq.3 of the paper). The first element of the tuple API return value is RD.
3. **Explicitly defined the inverse operator in the ridge form of Eq. 1** (`Xᵀ Ŷ (ŶᵀŶ + εI)⁻¹`). The old implementation used `Xᵀ (Ŷᵀ)⁺`, which matches in the limit as `ε→0`, but explicitly including `ε` is more faithful to the paper and more stable.
4. **Stabilization of the optimizer**: Changed to the equilibrium gradient normalized by the number of rules plus a damping step. The old implementation tended to diverge with a fixed learning rate (failed on Titanic with RS ≈ 0.007), but after the fix, RS monotonically increases with respect to `β` and then saturates, maintaining the trade-off with the proximity term (suppressing the collapse of `B`).
5. Dependency Reduction: Removed mandatory dependencies on `sklearn` (standardization) and `networkx` (plotting) (standardized using `numpy` and using pure `matplotlib` for plotting). `scipy` is optional only for sparse inputs. This is for reproducibility and portability.

### Connection to Future Extensions (Project Policy)

A framework that re-constrains "form" starting from the inverse operator `A†` can be naturally extended to Graph-AIME (graph-structured antecedents), Prescriptive AI (antecedent → intervention), and QAIME (rules with uncertainty). It can be generalized to higher-order conjunctions and graph constraints simply by substituting the set of antecedents for `R_rule` (this implementation is limited to pairwise).

---

## 6. Limitations (Based on the Paper)

- Evaluation is limited to 2 datasets and 3 classical models. No direct comparisons with rule learning/extraction methods (SBRL, IDS, CORELS, QCBA) have been conducted.
- Rules are limited to monotonicity, thresholds, and pairwise AND (higher-order conjunctions, disjunctions, and XOR-like interactions are out of scope).
- Does not support dense continuous modalities (such as raw images). Preprocessing, such as feature grouping or embedding-level constraints, is required for application.
- No verification of "auditability" for end-users has been conducted.
