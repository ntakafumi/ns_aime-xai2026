"""
ns_aime_xai2026.core
============

Core mathematics of NS-AIME (Sections 3.1 and 3.3 of the paper).

Theory <-> code map
-------------------
* ``NSAIME._inverse_operator`` -> Eq. (1)  A_dagger = X^T Yhat (Yhat^T Yhat + eps I)^-1
* ``NSAIME.fit``               -> Eq. (1) + Section 3.2 rule induction
* ``NSAIME.optimize``          -> Eq. (2)  min_B ||B - A_dagger||_F^2
                                            + alpha R_mono(B; M) + beta R_rule(B; T, P)

The optimization in Eq. (2) is convex (a quadratic proximity term plus convex
hinge penalties), so projected/sub-gradient descent converges to the global
optimum.  We keep the raw inverse surrogate ``A_dagger_`` immutable so that the
dual-reporting metrics (Eq. 3, 4) remain valid.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .utils import induce_rules, _is_sparse, _antecedent_mask


def _to_dense(X) -> np.ndarray:
    if _is_sparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


class NSAIME:
    """Neuro-Symbolic Approximate Inverse Model Explanations.

    Parameters
    ----------
    alpha : float
        Weight of the monotonicity penalty ``R_mono`` (Eq. 2).
    beta : float
        Weight of the rule (threshold + pairwise) penalty ``R_rule`` (Eq. 2).
    gamma : float
        Margin used by the rule penalty / RS: a positive-rule score must reach
        ``+gamma`` (and a negative-rule score ``-gamma``) to count as satisfied.
    top_k : int
        Number of important features kept per class during rule induction.
    lr : float
        Learning rate for the sub-gradient descent solver.
    steps : int
        Number of gradient steps.
    epsilon : float
        Ridge term ``eps`` in Eq. (1) for numerical stability.
    normalize : bool
        If True, features are standardized (zero mean, unit variance) internally
        before computing ``A_dagger`` and the explanation scores.  Rule display
        thresholds are still reported in the original feature units.
    include_pairwise : bool
        If True, the pairwise-AND co-activation rules also contribute to
        ``R_rule`` during optimization.
    max_instances : int
        Sub-sample size used when evaluating the rule penalty (for speed).
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.0,
        gamma: float = 0.05,
        top_k: int = 8,
        lr: float = 0.02,
        steps: int = 800,
        epsilon: float = 1e-3,
        normalize: bool = True,
        include_pairwise: bool = True,
        max_instances: int = 1500,
        seed: int = 42,
        # accepted for backward compatibility (unused by the convex solver)
        use_huber: bool = False,
        delta: float = 1.0,
        max_iter_huber: int = 50,
        tol_huber: float = 1e-5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k
        self.lr = lr
        self.steps = steps
        self.epsilon = epsilon
        self.normalize = normalize
        self.include_pairwise = include_pairwise
        self.max_instances = max_instances
        self.rng = np.random.RandomState(seed)

        # fitted attributes
        self.A_dagger_: Optional[np.ndarray] = None   # raw inverse surrogate (Eq. 1)
        self.B_: Optional[np.ndarray] = None          # re-constrained matrix (Eq. 2)
        self.rules_ = None
        self.feature_names_: Optional[list] = None
        self.class_names_: Optional[list] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.loss_history_: list = []

    # ------------------------------------------------------------------ #
    # internal: standardization (numpy only, no sklearn dependency)
    # ------------------------------------------------------------------ #
    def _standardize(self, X_dense: np.ndarray, fit: bool) -> np.ndarray:
        if not self.normalize:
            return X_dense
        if fit:
            self.mean_ = X_dense.mean(axis=0)
            self.std_ = X_dense.std(axis=0)
            self.std_[self.std_ < 1e-12] = 1.0
        return (X_dense - self.mean_) / self.std_

    @staticmethod
    def _sanitize(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        a[~np.isfinite(a)] = 0.0
        return np.clip(a, -1e6, 1e6)

    # ------------------------------------------------------------------ #
    # Eq. (1): deterministic inverse surrogate
    # ------------------------------------------------------------------ #
    def _inverse_operator(self, Xp: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
        """A_dagger = X^T Yhat (Yhat^T Yhat + eps I)^-1  (Eq. 1).

        ``Xp`` is the (already standardized) feature matrix; ``Yhat`` are the
        predicted class probabilities.  The k-th column of the result is the
        global contribution vector for class k.
        """
        Yhat = self._sanitize(Yhat)
        if Yhat.ndim == 1:
            Yhat = Yhat.reshape(-1, 1)
        m = Yhat.shape[1]
        gram = Yhat.T @ Yhat + self.epsilon * np.eye(m)   # (m, m), SPD
        A = Xp.T @ Yhat @ np.linalg.inv(gram)             # (d, m)
        return self._sanitize(A)

    # ------------------------------------------------------------------ #
    # fit: Eq. (1) + rule induction (Section 3.2)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        Yhat,
        feature_names: Sequence[str],
        y_labels: Optional[np.ndarray] = None,
        class_names: Optional[Sequence[str]] = None,
        q: float = 0.5,
        normalize: Optional[bool] = None,
    ) -> "NSAIME":
        if normalize is not None:
            self.normalize = normalize

        X_orig = _to_dense(X)
        Xp = self._sanitize(self._standardize(X_orig, fit=True))
        Yhat = self._sanitize(Yhat)
        if Yhat.ndim == 1:
            Yhat = Yhat.reshape(-1, 1)

        # Eq. (1) - raw inverse surrogate (kept immutable from here on).
        self.A_dagger_ = self._inverse_operator(Xp, Yhat)

        # Predicted labels (argmax of the black-box output).
        if y_labels is None:
            y_labels = (
                np.zeros(Yhat.shape[0], dtype=int)
                if Yhat.shape[1] == 1
                else np.argmax(Yhat, axis=1)
            )
        y_labels = np.asarray(y_labels)

        self.feature_names_ = list(feature_names)
        m = self.A_dagger_.shape[1]
        self.class_names_ = (
            list(class_names) if class_names is not None
            else [f"class_{k}" for k in range(m)]
        )

        # Section 3.2 - induce rules.  A_dagger_ is passed read-only; thresholds
        # for math live in standardized space (Xp), display thresholds in X_orig.
        self.rules_ = induce_rules(
            self.A_dagger_,
            Xp,
            y_labels,
            self.feature_names_,
            top_k=self.top_k,
            q=q,
            X_orig=X_orig,
        )
        # store the processed matrix used for optimization/metrics
        self._Xp_cache = Xp
        return self

    # ------------------------------------------------------------------ #
    # optimize: Eq. (2) convex re-constraint
    # ------------------------------------------------------------------ #
    def optimize(self, X=None) -> np.ndarray:
        """Solve Eq. (2) by sub-gradient descent, returning the matrix B.

        J(B) = ||B - A_dagger||_F^2
               + alpha * sum_{(j,k): M!=0} max(0, -M_jk B_jk)        # R_mono
               + beta  * sum_rules sum_{i in antecedent} margin_hinge # R_rule

        The proximity term keeps B close to the raw inverse surrogate; the hinge
        terms push the per-class explanation scores ``x_i . B[:,k]`` to the
        rule-consistent side of the margin for instances that satisfy the rule
        antecedent.  All terms are convex in B.
        """
        if self.A_dagger_ is None:
            raise ValueError("Call fit() before optimize().")

        if X is None:
            Xp = self._Xp_cache
        else:
            Xp = self._sanitize(self._standardize(_to_dense(X), fit=False))

        # sub-sample instances for the rule penalty (speed)
        n = Xp.shape[0]
        if n > self.max_instances:
            sel = self.rng.choice(n, self.max_instances, replace=False)
            Xs = Xp[sel]
        else:
            Xs = Xp
        n_s = Xs.shape[0]

        A = self.A_dagger_
        B = A.copy()                 # start from the raw inverse surrogate
        m = B.shape[1]
        gamma = self.gamma
        self.loss_history_ = []

        # pre-extract antecedent masks (constant in B) for every rule
        single_rules = []  # (k, j, sign, mask)
        pair_rules = []    # (k, sign, mask)
        rule_count = np.ones(m)  # #active rules per class, for balancing R_rule
        for k in range(m):
            cnt = 0
            for r in self.rules_[k]:
                if r.get("type", "single") == "single":
                    msk = _antecedent_mask(Xs[:, r["feature_idx"]], r)
                    if msk.any():
                        single_rules.append((k, r["feature_idx"], r["sign"], msk))
                        cnt += 1
                elif self.include_pairwise:
                    j1, j2 = r["features"]
                    a1 = (Xs[:, j1] > 0) if r["presence"][0] else (
                        Xs[:, j1] >= r["thresholds"][0] if r["signs"][0] > 0
                        else Xs[:, j1] <= r["thresholds"][0])
                    a2 = (Xs[:, j2] > 0) if r["presence"][1] else (
                        Xs[:, j2] >= r["thresholds"][1] if r["signs"][1] > 0
                        else Xs[:, j2] <= r["thresholds"][1])
                    msk = a1 & a2
                    if msk.any():
                        pair_rules.append((k, r["sign"], msk))
                        cnt += 1
            rule_count[k] = max(1, cnt)

        for step in range(self.steps):
            # decaying step size keeps the convex sub-gradient descent stable
            lr_t = self.lr / (1.0 + 2.0 * step / max(1, self.steps))
            grad = 2.0 * (B - A)                       # d/dB of proximity term
            loss = float(np.sum((B - A) ** 2))

            # --- R_mono: hinge on sign violations -------------------------- #
            for k in range(m):
                for r in self.rules_[k]:
                    if r.get("type", "single") != "single":
                        continue
                    j, s = r["feature_idx"], r["sign"]
                    if s != 0 and s * B[j, k] < 0:
                        grad[j, k] += self.alpha * (-s)
                        loss += self.alpha * (-s * B[j, k])

            # --- R_rule: margin hinge on explanation scores ---------------- #
            # Each rule contributes the *mean* feature vector over its violating
            # antecedent instances; the per-class total is divided by the number
            # of rules so that the proximity <-> rule trade-off stays balanced
            # (a single dominant rule cannot collapse the whole column).
            if self.beta > 0 and (single_rules or pair_rules):
                S = Xs @ B                              # (n_s, m) scores
                acc = np.zeros_like(B)                  # (d, m) accumulated R_rule grad
                for (k, j, s, msk) in single_rules:
                    sc = S[msk, k]
                    na = msk.sum()
                    if s > 0:
                        viol = sc < gamma
                        if viol.any():
                            acc[:, k] += -Xs[msk][viol].sum(0) / na
                            loss += self.beta * np.sum(gamma - sc[viol]) / na
                    else:
                        viol = sc > -gamma
                        if viol.any():
                            acc[:, k] += Xs[msk][viol].sum(0) / na
                            loss += self.beta * np.sum(sc[viol] + gamma) / na
                for (k, s, msk) in pair_rules:
                    sc = S[msk, k]
                    na = msk.sum()
                    viol = sc < gamma                   # AND rules promote class
                    if viol.any():
                        acc[:, k] += -Xs[msk][viol].sum(0) / na
                        loss += self.beta * np.sum(gamma - sc[viol]) / na
                grad += self.beta * acc / rule_count    # balance across rules

            grad = self._sanitize(grad)
            grad = np.clip(grad, -50.0, 50.0)           # safety against blow-up
            B = B - lr_t * grad
            self.loss_history_.append(loss)

        self.B_ = self._sanitize(B)
        return self.B_

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def fit_transform(self, X, Yhat, feature_names, **kw) -> np.ndarray:
        self.fit(X, Yhat, feature_names, **kw)
        return self.optimize()
