import numpy as np
from itertools import product


# Materialize the cartesian product only when it fits in a modest buffer.
# With U=6, models=6, L=3 the joint space is 6^6 * 3^6 ≈ 34M — never build it.
_MATERIALIZE_CAP = 100_000


class ActionSpace(object):

    def __init__(self, discvars, contexts):

        # Get the name of the parameters
        self._action_keys = list(discvars.keys())
        self._context_keys = list(contexts.keys())

        self._action_choices = [np.asarray(discvars[k]) for k in self._action_keys]
        self._action_cardinalities = [len(c) for c in self._action_choices]
        total = int(np.prod(self._action_cardinalities)) if self._action_cardinalities else 0
        self._joint_size = total

        if 0 < total <= _MATERIALIZE_CAP:
            self._allActions = np.array(list(product(*self._action_choices)))
        else:
            # On-the-fly sampling only; never allocate the full joint grid.
            self._allActions = None

        # preallocated memory for X and Y points
        self._context = np.empty(shape=(0, self.context_dim))
        self._action = np.empty(shape=(0, self.action_dim))
        self._context_action = np.empty(shape=(0, self.context_dim + self.action_dim))
        self._reward = np.empty(shape=(0))

    def __len__(self):
        assert len(self._action) == len(self._reward)
        assert len(self._action) == len(self._context)
        return len(self._reward)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def context(self):
        return self._context

    @property
    def action(self):
        return self._action

    @property
    def context_action(self):
        return self._context_action

    @property
    def reward(self):
        return self._reward

    @property
    def context_dim(self):
        return len(self._context_keys)

    @property
    def action_dim(self):
        return len(self._action_keys)

    @property
    def context_keys(self):
        return self._context_keys

    @property
    def action_keys(self):
        return self._action_keys

    @property
    def bounds(self):
        return self._bounds

    def action_to_array(self, action):
        try:
            assert set(action) == set(self._action_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(action)) +
                "not match the expected set of keys ({}).".format(self._action_keys)
            )
        return np.asarray([action[key] for key in self._action_keys])

    def context_to_array(self, context):
        try:
            assert set(context) == set(self._context_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(context)) +
                "not match the expected set of keys ({}).".format(self._context_keys)
            )
        return np.asarray([context[key] for key in self._context_keys])

    def array_to_action(self, x):
        try:
            assert len(x) == len(self._action_keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._action_keys))
            )
        # Discrete arms are integer-valued; cast so model/cell indexing works
        # even when candidates were sampled into a float matrix for the GP.
        out = {}
        for key, val, choices in zip(self._action_keys, x, self._action_choices):
            if np.issubdtype(np.asarray(choices).dtype, np.integer):
                out[key] = int(round(float(val)))
            else:
                out[key] = val
        return out

    def array_to_context(self, x):
        try:
            assert len(x) == len(self._context_keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._context_keys))
            )
        return dict(zip(self._context_keys, x))

    def register(self, context, action, reward):

        c = self.context_to_array(context)
        a = self.action_to_array(action)
        ca = np.concatenate([c.reshape(1, -1), a.reshape(1, -1)], axis=1)

        self._context = np.concatenate([self._context, c.reshape(1, -1)])
        self._action = np.concatenate([self._action, a.reshape(1, -1)])
        self._reward = np.concatenate([self._reward, [reward]])
        self._context_action = np.concatenate([self._context_action, ca.reshape(1, -1)])

    def sample_actions(self, k, rng=None):
        """Draw ``k`` joint actions by independent per-dimension sampling.

        Used when the cartesian product is too large to materialize. The private
        ``rng`` keeps candidate draws off the global NumPy stream.
        """
        rng = np.random if rng is None else rng
        cols = []
        for choices in self._action_choices:
            idx = rng.randint(0, len(choices), size=k)
            cols.append(np.asarray(choices)[idx])
        if not cols:
            return np.empty((k, 0))
        # Float matrix for GP / acquisition; array_to_action casts ints back.
        return np.column_stack([c.astype(np.float64, copy=False) for c in cols])

    def random_sample(self):
        if self._allActions is not None:
            rand_idx = np.random.randint(len(self._allActions))
            return self._allActions[rand_idx, :]
        return self.sample_actions(1)[0]

    def res(self):
        """Get all reward values found and corresponding parameters."""
        context = [dict(zip(self._context_keys, p)) for p in self.context]
        action = [dict(zip(self._action_keys, p)) for p in self.action]

        return [
            {"reward": r, "action": a, "context": c}
            for r, a, c in zip(self.reward, action, context)
        ]
