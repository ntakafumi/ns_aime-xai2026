# Rule-Guided Re-Constraint of Approximate Inverse Explanations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Research Only](https://img.shields.io/badge/License-Research_Only-red.svg)](#-license)

**NS-AIME** is a rule-guided re-constraint layer on top of **AIME**
(Approximate Inverse Model Explanations). **NS-AIME** transforms AIME's numerical explanation matrix into a rule-auditable **directed explanation graph**.Starting from AIME's deterministic global attribution matrix `A†`, it induces monotonicity, threshold, and pairwise-AND rules from data, then solves a convex optimization problem that stays close to `A†` while encouraging rule satisfaction. The constrained matrix `B` is used to construct a human-readable **directed explanation graph**, in which features, thresholds, and pairwise-AND relations are organized as an auditable explanatory structure. NS-AIME reports the raw AIME explanation and the constrained graph-based explanation **together** (dual reporting), so large divergence is treated as an *audit signal*, never as a silent replacement.

Reference implementation for

NS-AIME:
Rule-Guided Re-Constraint of Approximate Inverse Explanations
(XAI 2026 Late-Breaking Results)

> **Paper:** T. Nakanishi, *NS-AIME: Rule-Guided Re-Constraint of Approximate
> Inverse Explanations*, 4th World Conference on eXplainable AI
> (Late-breaking), Fortaleza, Brazil, 2026.
>
> See [`ALGORITHM.md`](ALGORITHM.md) for the full algorithm, the theory↔code map,
> and the assumptions made by this implementation.

## ✨ Key Features

* **Paper-faithful core:** `A† = Xᵀ Ŷ (ŶᵀŶ + εI)⁻¹` (Eq. 1) kept **immutable**, a
  convex re-constraint (Eq. 2), and dual-reporting metrics **RD / GA / RS**
  (Eq. 3, 4 + §4.1).
* **Data-Driven Rule Induction:** monotonicity signs, thresholds (incl. *presence*
  rules for sparse TF–IDF) and pairwise co-activations — no human input.
* **Catchy AIME-style visualizations:** a static radial logic graph
  (`plot_logic_graph`), a dual-report bar chart (`plot_dual_report`), and a
  **self-contained interactive HTML** graph with draggable nodes and animated
  directional flow (`render_logic_graph_html`) — no networkx / pyvis / d3.
* **Lightweight & reproducible:** core depends only on `numpy` + `matplotlib`
  (`scipy` optional for sparse inputs).
* **Model Agnostic:** any classifier that outputs probabilities.

## 📦 Installation

### From Source (Development)
```bash
git clone [https://github.com/your-username/ns-aime-xai2026.git](https://github.com/your-username/ns-aime-xai2026.git)
cd ns-aime-xai2026
pip install .
```

### From Built Package (.tar.gz)
If you have built the distribution package:
```bash
pip install dist/ns_aime_xai2026-0.2.0.tar.gz
```

## 🚀 Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from ns_aime_xai2026 import (NSAIME, compute_metrics,
                     plot_logic_graph, plot_dual_report, render_logic_graph_html)

# X (n×d), y, feature_names already loaded
model = RandomForestClassifier().fit(X_train, y_train)
Yhat = model.predict_proba(X_train)                 # black-box output

# 1. AIME inverse surrogate (Eq.1) + rule induction (§3.2)
ex = NSAIME(alpha=1.0, beta=3.0, gamma=0.05, top_k=8, normalize=True)
ex.fit(X_train, Yhat, feature_names=feature_names,
       class_names=["not_survived", "survived"])

# 2. Convex re-constraint (Eq.2) -> matrix B
B = ex.optimize()

# 3. Dual-reporting metrics (Eq.3 RD, Eq.4 GA, RS)
raw = compute_metrics(ex.A_dagger_, ex.A_dagger_, ex._Xp_cache, ex.rules_)  # raw AIME
ns  = compute_metrics(B,            ex.A_dagger_, ex._Xp_cache, ex.rules_)  # NS-AIME
print(f"RS {raw['rs']:.3f} -> {ns['rs']:.3f} | RD={ns['rd']:.2f} GA={ns['ga']:+.2f}")

# 4a. Catchy static logic graph (AIME style)
plot_logic_graph(B, ex.rules_, feature_names, class_index=1, class_name="survived")

# 4b. Dual-report bars (raw A† vs NS-AIME B) — large gap = audit signal
plot_dual_report(ex.A_dagger_, B, feature_names, 1, "survived", rd=ns["rd"], ga=ns["ga"])

# 4c. Interactive HTML (drag nodes, animated flow, hover tooltips)
render_logic_graph_html(B, ex.A_dagger_, ex.rules_, feature_names,
                        ["not_survived", "survived"],
                        metrics=ns, raw_metrics=raw, path="logic_graph.html")
```

Full runnable demo (Titanic + 20 Newsgroups): [`examples/ns_aime_demo.ipynb`](examples/ns_aime_demo.ipynb).

## 📊 Visualizations
* **Coral edge** — feature *promotes* the class (monotonicity `M = +1`).
* **Indigo edge** — feature *inhibits* the class (`M = −1`).
* **Violet `&` node** — pairwise AND co-activation rule.
* **Threshold chip** — the induced condition (`≥ t`, `≤ t`, or `present`).
* Edge direction always flows **feature → class** (the explanation answers
  "which inputs drive this output"); the interactive HTML animates that flow.

## 📚 Citation
If you use this software in your research, please cite our paper:
```
@inproceedings{nakanishi2026nsaime,
  title     = {NS-AIME: Rule-Guided Re-Constraint of Approximate Inverse Explanations},
  author    = {Nakanishi, Takafumi},
  booktitle = {Late-breaking Work, Demos and Doctoral Consortium Joint Proceedings, co-located with the 4th World Conference on eXplainable Artificial Intelligence (xAI 2026)},
  year      = {2026},
  address   = {Fortaleza, Brazil},
  publisher = {CEUR-WS.org},
  note      = {Accepted for publication and poster presentation}
}
```

## 📝 License

**NS-AIME** is free for **non-commercial research and educational purposes**.

For **commercial use** (e.g., integrating into products, internal business use), a separate commercial license is required. Please contact the author for licensing inquiries:

📧 **Contact:** takafumi@eigenbeats.com

See the [LICENSE](LICENSE) file for full details.
