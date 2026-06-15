"""
ns_aime_xai2026.utils
=============

Rule induction and dual-reporting metrics for NS-AIME.

This module is a faithful implementation of Sections 3.2 and 3.4 of

    T. Nakanishi, "NS-AIME: Rule-Guided Re-Constraint of Approximate Inverse
    Explanations", 4th World Conference on eXplainable AI (Late-breaking), 2026.

Theory <-> code map
-------------------
* ``induce_rules``      -> Section 3.2 (monotonicity, thresholds, pairwise AND).
* ``compute_metrics``   -> Section 3.4, Eq. (3) RD, Eq. (4) GA, plus Rule
                           Satisfaction (RS, Section 4.1).

Design note (paper compliance)
-------------------------------
The *raw* inverse surrogate ``A_dagger`` (Eq. 1) is treated as **immutable**.
Rule induction only *reads* ``A_dagger``; it never overwrites it.  This is
essential for the dual-reporting protocol (Section 3.4): RD and GA are only
meaningful if ``A_dagger`` still equals the unconstrained inverse surrogate.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence
import numpy as np


# --------------------------------------------------------------------------- #
# Small helpers (keep scipy optional: it is only needed for sparse inputs)
# --------------------------------------------------------------------------- #
def _is_sparse(X) -> bool:
    """True for a scipy.sparse matrix without importing scipy at module load."""
    return hasattr(X, "toarray") and not isinstance(X, np.ndarray)


def _col(X, j) -> np.ndarray:
    """Return column ``j`` of a dense or sparse matrix as a 1-D float array."""
    if _is_sparse(X):
        return np.asarray(X[:, j].todense()).ravel().astype(np.float64)
    return np.asarray(X[:, j], dtype=np.float64).ravel()


def _rows(X, mask) -> np.ndarray:
    """Return masked rows of a dense or sparse matrix as a dense array."""
    sub = X[mask]
    if _is_sparse(sub):
        return np.asarray(sub.todense(), dtype=np.float64)
    return np.asarray(sub, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Section 3.2 - Rule induction
# --------------------------------------------------------------------------- #
def induce_rules(
    A_dagger: np.ndarray,
    X,
    y_labels: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 8,
    q: float = 0.5,
    corr_threshold: float = 0.05,
    and_corr_threshold: float = 0.15,
    presence_zero_frac: float = 0.5,
    X_orig=None,
) -> List[List[Dict[str, Any]]]:
    """Induce monotonicity, threshold and pairwise-AND rules from data.

    Implements Section 3.2.  For every class ``k`` we look only at the
    ``top_k`` features with the largest ``|A_dagger[:, k]|`` ("sufficiently
    important features"), and induce:

    * **Monotonicity** ``M_jk in {-1, 0, +1}`` (sign label).  The sign is taken
      from the empirical correlation between feature ``j`` and the class
      indicator ``1[y = k]`` (data-driven); it falls back to ``sign(A_dagger)``
      when the correlation is too weak to be informative.
    * **Threshold** ``t_jk`` estimated from the feature values of instances
      predicted as class ``k`` (median quantile ``q`` for continuous features,
      the binary mid-point for two-valued features, and *presence* ``t = 0`` for
      sparse features such as TF-IDF).
    * **Pairwise co-activation** ``(x_j1 >= t1) AND (x_j2 >= t2) => k`` for pairs
      of high-contributing features whose joint activation correlates with the
      class (direction flipped when monotonicity is negative).

    Parameters
    ----------
    A_dagger : (d, m) ndarray
        Raw inverse surrogate (Eq. 1).  **Read only** - never modified here.
    X : (n, d) array or scipy.sparse
        Feature matrix used for threshold/sign estimation.  This must live in
        the *same space* in which the explanation scores ``X @ B`` are later
        evaluated (the standardized space, when ``NSAIME(normalize=True)``).
    y_labels : (n,) ndarray
        Predicted class label per instance (``argmax`` of the model output).
    feature_names : sequence of str
    top_k : int
        Number of important features per class.
    q : float
        Quantile used as the threshold for continuous features.
    corr_threshold : float
        Minimum |corr| for a *data-driven* sign; below this we fall back to the
        sign of ``A_dagger``.
    and_corr_threshold : float
        Minimum joint-activation correlation to keep a pairwise AND rule.
    presence_zero_frac : float
        If at least this fraction of a feature's values are zero, the feature is
        treated as sparse and a *presence* rule (``x_j > 0``) is used.
    X_orig : optional array
        Feature matrix in the *original* (un-standardized) units, used only to
        compute human-readable display thresholds.  If ``None``, ``X`` is used.

    Returns
    -------
    list of length m
        ``rules[k]`` is a list of rule dicts.  Single-feature rules have
        ``type='single'``; pairwise rules have ``type='and'``.
    """
    A_dagger = np.asarray(A_dagger, dtype=np.float64)
    d, m = A_dagger.shape
    n = y_labels.shape[0]

    # Make sure we have a name for every feature.
    names = list(feature_names)
    if len(names) < d:
        names += [f"feature_{i}" for i in range(len(names), d)]

    X_disp = X if X_orig is None else X_orig

    rules: List[List[Dict[str, Any]]] = []

    for k in range(m):
        coefs = np.nan_to_num(A_dagger[:, k])
        if np.allclose(coefs, 0.0):
            rules.append([])
            continue

        kk = int(min(top_k, d))
        top_idx = np.argsort(np.abs(coefs))[-kk:][::-1]  # most important first

        y_target = (y_labels == k).astype(np.float64)
        class_mask = (y_labels == k)
        if class_mask.sum() == 0:
            class_mask = np.ones(n, dtype=bool)

        k_rules: List[Dict[str, Any]] = []

        for j in top_idx:
            full = _col(X, j)
            if full.std() < 1e-12:
                continue  # constant feature carries no rule

            full_disp = _col(X_disp, j)
            uniq = np.unique(full_disp)
            zero_frac = float(np.mean(np.isclose(full, 0.0)))

            # --- monotonicity sign M_jk (data-driven, fall back to A_dagger) ---
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(full, y_target)[0, 1]
            if np.isfinite(corr) and abs(corr) > corr_threshold:
                sign = float(np.sign(corr))
            else:
                sign = float(np.sign(coefs[j]))
            if sign == 0.0:
                sign = 1.0

            # --- threshold t_jk (math space) + display threshold (orig units) --
            cls_vals = full[class_mask]
            cls_vals_disp = full_disp[class_mask]
            presence = False
            if len(uniq) <= 2:
                # binary feature -> mid-point (e.g. {0,1} -> 0.5; std-space too)
                thr = float(np.mean(np.unique(full)))
                thr_disp = float(np.mean(uniq))
            elif zero_frac >= presence_zero_frac:
                # sparse feature (e.g. TF-IDF) -> presence rule x_j > 0
                presence = True
                sign = 1.0  # presence implies a positive (promoting) antecedent
                thr = 0.0
                thr_disp = 0.0
            else:
                thr = float(np.quantile(cls_vals, q))
                thr_disp = float(np.quantile(cls_vals_disp, q))

            k_rules.append(
                {
                    "type": "single",
                    "feature_idx": int(j),
                    "name": names[j],
                    "sign": sign,                  # M_jk in {-1,+1}
                    "threshold": thr,              # used by optimizer / RS (math space)
                    "threshold_display": thr_disp, # human-readable (orig units)
                    "presence": presence,
                    "weight": float(coefs[j]),     # raw A_dagger weight (for ranking)
                }
            )

        # --- pairwise AND rules among the strongest single features ----------
        pair_pool = list(top_idx[: min(5, len(top_idx))])
        singles = {r["feature_idx"]: r for r in k_rules if r["type"] == "single"}
        for a in range(len(pair_pool)):
            for b in range(a + 1, len(pair_pool)):
                j1, j2 = int(pair_pool[a]), int(pair_pool[b])
                if j1 not in singles or j2 not in singles:
                    continue
                r1, r2 = singles[j1], singles[j2]
                a1 = _antecedent_mask(_col(X, j1), r1)
                a2 = _antecedent_mask(_col(X, j2), r2)
                both = (a1 & a2).astype(np.float64)
                if both.std() < 1e-12:
                    continue
                with np.errstate(invalid="ignore"):
                    c = np.corrcoef(both, y_target)[0, 1]
                if np.isfinite(c) and c > and_corr_threshold:
                    k_rules.append(
                        {
                            "type": "and",
                            "features": [j1, j2],
                            "names": [names[j1], names[j2]],
                            "signs": [r1["sign"], r2["sign"]],
                            "thresholds": [r1["threshold"], r2["threshold"]],
                            "presence": [r1["presence"], r2["presence"]],
                            # AND promotes the class (direction already encoded
                            # in the per-feature signs / antecedents).
                            "sign": 1.0,
                            "importance": float(c),
                        }
                    )

        rules.append(k_rules)

    return rules


def _antecedent_mask(vals: np.ndarray, rule: Dict[str, Any]) -> np.ndarray:
    """Boolean mask of instances satisfying a single rule's antecedent."""
    if rule.get("presence", False):
        return vals > 0.0
    if rule["sign"] > 0:
        return vals >= rule["threshold"]
    return vals <= rule["threshold"]


# --------------------------------------------------------------------------- #
# Section 3.4 - Dual-reporting metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    B: np.ndarray,
    A_dagger: np.ndarray,
    X,
    rules: List[List[Dict[str, Any]]],
    gamma: float = 0.0,
    max_instances: int = 4000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute the NS-AIME dual-reporting metrics.

    Returns a dict with keys:

    * ``rd``       - Relative Deviation, Eq. (3): ``||B - A_dagger||_F / ||A_dagger||_F``
                     (lower = closer to raw AIME).  *Not* bounded by 1.
    * ``ga``       - Global Alignment, Eq. (4): mean per-class cosine similarity
                     between columns of ``B`` and ``A_dagger`` (higher = better
                     directional agreement; can be negative = audit flag).
    * ``rs``       - Rule Satisfaction (Section 4.1): fraction of
                     (rule, antecedent-active instance) pairs whose explanation
                     score ``x . B[:, k]`` has the rule-consistent sign/margin.
    * ``sparsity`` - fraction of near-zero entries in ``B``.

    Passing ``B = A_dagger`` yields the *raw* AIME metrics (RS(raw)), which is
    exactly what the dual-reporting protocol compares against.
    """
    B = np.nan_to_num(np.asarray(B, dtype=np.float64))
    A_dagger = np.nan_to_num(np.asarray(A_dagger, dtype=np.float64))
    m = B.shape[1]

    # Eq. (3) Relative Deviation
    denom = np.linalg.norm(A_dagger) + 1e-12
    rd = float(np.linalg.norm(B - A_dagger) / denom)

    # Eq. (4) Global Alignment (mean cosine over classes)
    cos = []
    for k in range(m):
        u, v = B[:, k], A_dagger[:, k]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cos.append(float(u @ v / (nu * nv)) if nu > 1e-12 and nv > 1e-12 else 0.0)
    ga = float(np.mean(cos)) if cos else 0.0

    # Rule Satisfaction
    n = X.shape[0]
    if n > max_instances:
        idx = np.random.RandomState(seed).choice(n, max_instances, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X
    if _is_sparse(X_eval):
        S = np.asarray(X_eval.dot(B))
    else:
        S = np.asarray(X_eval, dtype=np.float64) @ B

    total = 0
    satisfied = 0
    for k in range(m):
        for r in rules[k]:
            if r.get("type", "single") == "single":
                vals = _col(X_eval, r["feature_idx"])
                mask = _antecedent_mask(vals, r)
            else:  # pairwise AND
                m1 = _col(X_eval, r["features"][0]) > 0.0 if r["presence"][0] \
                    else _side_mask(_col(X_eval, r["features"][0]), r["signs"][0], r["thresholds"][0])
                m2 = _col(X_eval, r["features"][1]) > 0.0 if r["presence"][1] \
                    else _side_mask(_col(X_eval, r["features"][1]), r["signs"][1], r["thresholds"][1])
                mask = m1 & m2
            num = int(mask.sum())
            if num == 0:
                continue
            s = S[mask, k]
            if r.get("sign", 1.0) > 0:
                satisfied += int(np.sum(s >= gamma))
            else:
                satisfied += int(np.sum(s <= -gamma))
            total += num

    rs = float(satisfied / total) if total > 0 else 1.0
    sparsity = float(np.mean(np.abs(B) < 1e-3))

    return {"rd": rd, "ga": ga, "rs": rs, "sparsity": sparsity}


def _side_mask(vals, sign, thr):
    return vals >= thr if sign > 0 else vals <= thr


def calculate_metrics(B, A_dagger, X, rules, gamma: float = 0.0):
    """Backward-compatible tuple API: ``(RD, GA, sparsity, RS)``.

    Note: element 0 is now **RD** (Relative Deviation, Eq. 3), which the paper
    renamed from the misleading term "fidelity" - lower is better.
    """
    d = compute_metrics(B, A_dagger, X, rules, gamma=gamma)
    return d["rd"], d["ga"], d["sparsity"], d["rs"]
