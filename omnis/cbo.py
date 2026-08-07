import warnings
import numpy as np
from omnis.action_space import ActionSpace
from omnis.fast_gp import FastGaussianProcessRegressor
from omnis.util import acq_max, acq_max_sample

from sklearn.gaussian_process import GaussianProcessRegressor


class ContextualBayesianOptimization():
    def __init__(self, all_actions_dict, contexts, kernel, noise=1e-6, points=[], rewards=[],
                 init_random=3, gp_burn_in=25, n_restarts_optimizer=2, use_fast_gp=True):
        
        self._space = ActionSpace(all_actions_dict, contexts)
        self.init_random = init_random
        # Private RNG for candidate subsampling: keeps the global RNG stream
        # (tasks, noise realizations) identical to the exhaustive-search version
        self._candidate_rng = np.random.RandomState(2024)
        # gp_burn_in <= 0: always L-BFGS ARD every fit (full joint CBO cost).
        # gp_burn_in > 0: optimize for the first N observations, then freeze.
        # Per-user Causal/UCB keep burn-in freeze; CTO passes 0 for full cost.
        self.gp_burn_in = int(gp_burn_in)
        self._gp_hypers_frozen = False
        
        if len(points) > 0:
            gp_hyp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=noise,
                normalize_y=True,
                n_restarts_optimizer=max(5, int(n_restarts_optimizer)))
            
            print('Optimizing kernel hyperparameters....')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_hyp.fit(points, rewards)
            print('Done!')
            
            opt_hyp = gp_hyp.kernel_.get_params()
            kernel.set_params(**opt_hyp)
            optimizer = None
            self._gp_hypers_frozen = True
        else:
            # warnings.warn('Kernel hyperparameters will be computed during the optimization.')
            optimizer = 'fmin_l_bfgs_b'

        # FastGP: BLAS predict for light per-user CBO (Causal/UCB). CTO sets
        # use_fast_gp=False so joint K-candidate scoring keeps centralized cost.
        gp_cls = FastGaussianProcessRegressor if use_fast_gp else GaussianProcessRegressor
        self._gp = gp_cls(
            kernel=kernel,
            alpha=noise,
            normalize_y=True,
            optimizer=optimizer,
            n_restarts_optimizer=int(n_restarts_optimizer))

    @property
    def space(self):
        return self._space

    @property
    def res(self):
        return self._space.res()

    def register(self, context, action, reward):
        """Expect observation with known reward"""
        self._space.register(context, action, reward)

    def predict_mean(self, context, action):
        """Posterior mean at (context, action) if the GP is already fitted.

        Returns None during the random init phase or if fit has not run yet.
        Used for optional reward prediction-error logging (UCB/DTS/CTO).
        """
        if len(self._space) < self.init_random:
            return None
        if not hasattr(self._gp, "X_train_") or self._gp.X_train_ is None:
            return None
        try:
            c = self._space.context_to_array(context).reshape(1, -1)
            a = self._space.action_to_array(action).reshape(1, -1)
            ca = np.concatenate([c, a], axis=1)
            mu = self._gp.predict(ca)
            return float(np.asarray(mu).ravel()[0])
        except Exception:
            return None

    def array_to_context(self, context):
        return self._space.array_to_context(context)
    
    def action_to_array(self, action):
        return self._space.action_to_array(action)

    def context_to_array(self, context):
        return self._space.context_to_array(context)

    def _fit_gp(self):
        """Fit posterior; optional burn-in freeze, else full L-BFGS every slot."""
        n = len(self._space)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Positive burn-in: one last L-BFGS at N, then Cholesky-only fits.
            if (self.gp_burn_in > 0
                    and (not self._gp_hypers_frozen)
                    and n >= self.gp_burn_in):
                self._gp.optimizer = 'fmin_l_bfgs_b'
                self._gp.fit(self._space.context_action, self._space.reward)
                self._gp.kernel = self._gp.kernel_
                self._gp.optimizer = None
                self._gp_hypers_frozen = True
            else:
                # gp_burn_in <= 0 keeps optimizer='fmin_l_bfgs_b' every call.
                self._gp.fit(self._space.context_action, self._space.reward)

    def suggest(self, context, utility_function, max_candidates=None,
                score_scale=1.0, score_offset=None):
        """Most promising point to probe next"""
        assert len(context) == self._space.context_dim
        context = self._space.context_to_array(context)
        if len(self._space) < self.init_random:
            return self._space.array_to_action(self._space.random_sample())

        self._fit_gp()

        space = self._space
        use_onthefly = (
            max_candidates is not None
            and (space._allActions is None
                 or len(space._allActions) > max_candidates)
        )

        if use_onthefly and space._allActions is None:
            # Never materialize the joint grid; sample K candidates directly.
            suggestion = acq_max_sample(
                ac=utility_function.utility,
                gp=self._gp,
                context=context,
                action_sampler=space.sample_actions,
                max_candidates=max_candidates,
                rng=self._candidate_rng,
                score_scale=score_scale,
                score_offset=score_offset)
        else:
            # Finding argmax of the acquisition function (materialized pool).
            suggestion = acq_max(
                ac=utility_function.utility,
                gp=self._gp,
                all_discr_actions=space._allActions,
                context=context,
                max_candidates=max_candidates,
                rng=self._candidate_rng,
                score_scale=score_scale,
                score_offset=score_offset)

        return self._space.array_to_action(suggestion)
