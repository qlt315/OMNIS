import numpy as np
from scipy.linalg import solve_triangular


class ResidualGP:
    """Gaussian process over the residual r(x) = f(x) - prior_mean(x) with
    exact online posterior updates via incremental Cholesky factorization.

    Kernel: anisotropic Matern-3/2 with fixed hyperparameters, matching the
    kernel family used by the conference-version GP-UCB agents. Each `add`
    costs O(n) (one triangular row extension into a preallocated buffer); the
    O(n^2) posterior solve is deferred to the next `predict` call, so it runs
    once per slot rather than once per observation. No per-slot hyperparameter
    optimization is performed, which removes the O(t^3) L-BFGS refits of the
    original implementation.

    With an empty (uninformative) prior and ``_n==0``, predictive std is
    inflated so UCB/TS explores aggressively until observations arrive.
    """

    def __init__(self, prior_mean_fn, length_scales, signal_var=2.5e-3, noise_var=1e-4,
                 jitter=1e-8, init_capacity=64, empty_prior_std=1.0, max_obs=600):
        self.prior_mean_fn = prior_mean_fn
        self.length_scales = np.asarray(length_scales, dtype=float)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.jitter = float(jitter)
        # Large std when no observations → Acc UCB/TS explores (乱搞).
        self.empty_prior_std = float(max(
            empty_prior_std, 10.0 * np.sqrt(max(self.signal_var, 1e-12))))
        # Cap observations (sliding window) so 300-slot runs stay O(max_obs^2).
        self.max_obs = int(max_obs) if max_obs else 0

        self._n = 0
        self._cap = init_capacity
        dim = len(self.length_scales)
        self._X = np.zeros((self._cap, dim))
        self._y_res = np.zeros(self._cap)
        self._L = np.zeros((self._cap, self._cap))
        self._alpha = np.empty(0)
        self._alpha_dirty = True

    def _ensure_capacity(self, needed):
        if needed <= self._cap:
            return
        new_cap = max(2 * self._cap, needed)
        self._X = np.vstack([self._X, np.zeros((new_cap - self._cap, self._X.shape[1]))])
        self._y_res = np.append(self._y_res, np.zeros(new_cap - self._cap))
        L_new = np.zeros((new_cap, new_cap))
        L_new[:self._cap, :self._cap] = self._L
        self._L = L_new
        self._cap = new_cap

    def _kernel(self, X1, X2):
        d = (X1[:, None, :] - X2[None, :, :]) / self.length_scales
        r = np.sqrt(np.sum(d * d, axis=-1))
        sqrt3_r = np.sqrt(3.0) * r
        return self.signal_var * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)

    def add(self, x, y_obs):
        x = np.asarray(x, dtype=float).ravel()
        res = float(y_obs - self.prior_mean_fn(x))
        n = self._n
        self._ensure_capacity(n + 1)

        k_diag = self.signal_var + self.noise_var + self.jitter
        if n == 0:
            self._L[0, 0] = np.sqrt(k_diag)
        else:
            L = self._L[:n, :n]
            k_vec = self._kernel(self._X[:n], x[None, :]).ravel()
            l_new = solve_triangular(L, k_vec, lower=True)
            self._L[n, :n] = l_new
            self._L[n, n] = np.sqrt(max(k_diag - l_new @ l_new, self.jitter))

        self._X[n] = x
        self._y_res[n] = res
        self._n = n + 1
        self._alpha_dirty = True

    def maybe_trim(self):
        """Drop oldest obs if past ``max_obs`` (call once per slot batch)."""
        # Rebuild only after growing ~20% past the cap (not every add).
        if self.max_obs > 0 and self._n > int(self.max_obs * 1.2):
            self._rebuild_window(self.max_obs)

    def _rebuild_window(self, keep):
        """Keep the newest ``keep`` residuals and rebuild the Cholesky factor."""
        X = self._X[self._n - keep:self._n].copy()
        y_res = self._y_res[self._n - keep:self._n].copy()
        self._n = 0
        self._alpha = np.empty(0)
        self._alpha_dirty = True
        self._L[:] = 0.0
        for i in range(keep):
            x = X[i]
            res = float(y_res[i])
            n = self._n
            self._ensure_capacity(n + 1)
            k_diag = self.signal_var + self.noise_var + self.jitter
            if n == 0:
                self._L[0, 0] = np.sqrt(k_diag)
            else:
                L = self._L[:n, :n]
                k_vec = self._kernel(self._X[:n], x[None, :]).ravel()
                l_new = solve_triangular(L, k_vec, lower=True)
                self._L[n, :n] = l_new
                self._L[n, n] = np.sqrt(max(k_diag - l_new @ l_new, self.jitter))
            self._X[n] = x
            self._y_res[n] = res
            self._n = n + 1
        self._alpha_dirty = True

    def _sync_alpha(self):
        if not self._alpha_dirty:
            return
        n = self._n
        L = self._L[:n, :n]
        z = solve_triangular(L, self._y_res[:n], lower=True)
        self._alpha = solve_triangular(L.T, z, lower=False)
        self._alpha_dirty = False

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # Fast path: uninformative prior mean is identically 0 (Causal Acc GP).
        if self._n == 0:
            # Still evaluate one prior sample so custom priors stay correct.
            p0 = float(self.prior_mean_fn(X[0]))
            if p0 == 0.0 and len(X) > 1:
                prior = np.zeros(len(X))
            else:
                prior = np.array([self.prior_mean_fn(x) for x in X])
            return prior, np.full(len(X), self.empty_prior_std)

        self._sync_alpha()
        n = self._n
        k_star = self._kernel(self._X[:n], X)
        mean_res = k_star.T @ self._alpha
        v = solve_triangular(self._L[:n, :n], k_star, lower=True)
        var = self.signal_var - np.sum(v * v, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12))
        p0 = float(self.prior_mean_fn(X[0]))
        if p0 == 0.0:
            # Assume constant-zero prior (Acc-table prior banned).
            return mean_res, std
        prior = np.array([self.prior_mean_fn(x) for x in X])
        return prior + mean_res, std

    def sample(self, X):
        """Marginal Thompson sample from the posterior at X."""
        mean, std = self.predict(X)
        return mean + std * np.random.randn(len(mean))

    def __len__(self):
        return self._n
