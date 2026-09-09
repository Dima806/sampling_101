"""Linear congruential generators: one acceptable, one famously broken.

A linear congruential generator produces integers with

    x_{n+1} = (a * x_n + c) mod m

and turns them into floats in [0, 1) by dividing by m. It is the simplest generator worth
studying, and the difference between a good choice of (a, c, m) and a bad one is the whole
point of notebook 01.
"""

import numpy as np

# RANDU, shipped by IBM in the 1960s and used for a decade before anyone plotted it in 3-D.
# Its triples satisfy x_{n+2} = 6 * x_{n+1} - 9 * x_n (mod 2**31) exactly, so every triple
# lies on one of 15 parallel planes. The marginal distribution looks perfectly uniform.
RANDU: dict[str, int] = {"a": 65539, "c": 0, "m": 2**31}

# MINSTD (Park-Miller, Lewis-Goodman-Miller multiplier). A defensible minimal standard:
# full period, no low-order lattice structure at this scale.
MINSTD: dict[str, int] = {"a": 48271, "c": 0, "m": 2**31 - 1}


class LCG:
    """A linear congruential generator yielding floats in [0, 1).

    Parameters
    ----------
    a, c, m
        Multiplier, increment and modulus of ``x_{n+1} = (a * x_n + c) mod m``.
    seed
        Initial state, reduced modulo ``m``. For a multiplicative generator (``c == 0``)
        the seed must not reduce to zero, or the sequence collapses to all zeros.
    """

    def __init__(self, a: int, c: int, m: int, seed: int) -> None:
        if m <= 0:
            raise ValueError(f"modulus must be positive, got {m}")
        state = seed % m
        if c % m == 0 and state == 0:
            raise ValueError("a multiplicative LCG seeded at 0 produces only zeros")
        self.a = a
        self.c = c
        self.m = m
        self._state = state

    @classmethod
    def from_params(cls, params: dict[str, int], seed: int) -> "LCG":
        """Build a generator from one of the module-level parameter sets."""
        return cls(a=params["a"], c=params["c"], m=params["m"], seed=seed)

    @property
    def state(self) -> int:
        """Current integer state, so a stream can be reproduced exactly."""
        return self._state

    def random(self, n: int) -> np.ndarray:
        """Draw ``n`` floats in [0, 1) and advance the state.

        The recurrence is a scalar loop on purpose: it is the object being taught, and
        seeing one line per step is worth more here than a vectorised trick.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        out = np.empty(n, dtype=np.float64)
        x = self._state
        for i in range(n):
            x = (self.a * x + self.c) % self.m
            out[i] = x / self.m
        self._state = x
        return out
