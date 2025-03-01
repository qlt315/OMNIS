import warnings
import numpy as np

def acq_max(ac, gp, all_discr_actions, context):
    """
    A function to find the maximum of the acquisition function
    We evaluate all possible actions since we consider a discrete set of actions.
    """
    context_action = np.concatenate([np.tile(context, (len(all_discr_actions), 1)), all_discr_actions], axis=1)
    
    ys = ac(context_action, gp=gp)
    x_max = all_discr_actions[ys.argmax()]
    return x_max


import numpy as np
import warnings


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
