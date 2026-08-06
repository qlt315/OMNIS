import warnings
import numpy as np

def acq_max(ac, gp, all_discr_actions, context, max_candidates=None, rng=None,
            score_scale=1.0, score_offset=None):
    """
    A function to find the maximum of the acquisition function
    We evaluate all possible actions since we consider a discrete set of actions.
    If max_candidates is set and the action set is larger, a uniform random
    subset of that size is evaluated instead (standard random-search trade-off
    for large discrete spaces). rng decouples the subsampling draws from the
    global numpy RNG so downstream randomness (tasks, noise) is unaffected.

    When ``all_discr_actions`` is None, ``rng`` plus ``max_candidates`` must be
    provided together with ``action_sampler(k, rng) -> (k, action_dim)`` via the
    optional keyword (see ``acq_max_sample``). Prefer calling ``acq_max_sample``
    for on-the-fly pools.

    score_scale/score_offset implement the Lyapunov drift-plus-penalty scoring:
    the acquisition value is rescaled by the penalty weight V (score_scale) and
    the per-action analytic drift term is added (score_offset) before argmax.
    When score_offset is given together with subsampling, it must be indexed by
    the candidate position (caller passes offsets for the sampled subset).
    """
    if all_discr_actions is None:
        raise ValueError(
            "all_discr_actions is None; use acq_max_sample() for on-the-fly pools")

    if max_candidates is not None and len(all_discr_actions) > max_candidates:
        rng = np.random if rng is None else rng
        idx = rng.choice(len(all_discr_actions), max_candidates, replace=False)
        actions = all_discr_actions[idx]
        if score_offset is not None:
            score_offset = np.asarray(score_offset)[idx]
    else:
        actions = all_discr_actions

    context_action = np.concatenate([np.tile(context, (len(actions), 1)), actions], axis=1)

    ys = ac(context_action, gp=gp) * score_scale
    if score_offset is not None:
        ys = ys + score_offset
    x_max = actions[ys.argmax()]
    return x_max


def acq_max_sample(ac, gp, context, action_sampler, max_candidates, rng=None,
                   score_scale=1.0, score_offset=None):
    """Argmax acquisition over an on-the-fly pool of ``max_candidates`` actions.

    ``action_sampler(k, rng)`` returns a ``(k, action_dim)`` array. Used when the
    joint discrete space is too large to materialize (CTO with U users).
    """
    rng = np.random if rng is None else rng
    actions = action_sampler(int(max_candidates), rng)
    context_action = np.concatenate(
        [np.tile(context, (len(actions), 1)), actions], axis=1)
    ys = ac(context_action, gp=gp) * score_scale
    if score_offset is not None:
        ys = ys + np.asarray(score_offset)
    return actions[ys.argmax()]


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    Supports Upper Confidence Bound (UCB) and Thompson Sampling (TS).
    """

    def __init__(self, kind, beta_kind='const', beta_const=1):
        """
        Initialize the utility function.

        Args:
            kind (str): The type of acquisition function ('ucb' or 'ts').
            beta_kind (str): The type of beta parameter ('const' or 'theor').
            beta_const (float): Constant beta value for UCB.
        """
        self.beta_const = beta_const
        self.beta_val = 1
        self.t = 0
        self.delta = 0.01

        if kind not in ['ucb', 'ts']:
            raise NotImplementedError(f"The utility function {kind} has not been implemented.")
        else:
            self.kind = kind

        if beta_kind not in ['const', 'theor']:
            raise NotImplementedError(
                f"The beta function {beta_kind} has not been implemented, select 'const' or 'theor'.")
        else:
            self.beta_kind = beta_kind

    def update_params(self):
        """Update beta parameters for UCB."""
        self.t += 1
        if self.beta_kind == 'const':
            self.beta_val = self.beta_const
        elif self.beta_kind == 'theor':
            self.beta_val = 2 + 300 * self.t ** (33 / 34) * np.log10(self.t) * (np.log(self.t / self.delta) ** 3)

    def utility(self, x, gp):
        """
        Compute the acquisition function value.

        Args:
            x (np.array): The input points.
            gp (GaussianProcessRegressor): The trained Gaussian Process model.

        Returns:
            np.array: The acquisition function values.
        """
        self.update_params()
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.beta_val)
        elif self.kind == 'ts':
            return self._thompson_sampling(x, gp)

    @staticmethod
    def _ucb(x, gp, beta):
        """Compute the UCB acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        return mean + beta * std

    @staticmethod
    def _thompson_sampling(x, gp):
        """Compute the Thompson Sampling acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        return np.random.normal(mean, std)  # Sample from the posterior distribution
