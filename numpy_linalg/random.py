"""Wrapped version of module numpy.random
"""

import numpy.random as _pr
from . import wrappers as _wr
from ._lnarray import lnarray as _lnarray

__all__ = [
    'default_rng',
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'choice',
    'dirichlet',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'get_state',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'normal',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random_integers',
    'random_sample',
    'rayleigh',
    'seed',
    'set_state',
    'shuffle',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf',
    'ranf',
    'random',
    'sample',
]
_wrap = _wr.Wrappers(_lnarray, "numpy_linalg.random")

# =========================================================================
# Modern RNG
# =========================================================================


class LnGenerator(_wr.WrappedClass, wrappers=_wrap, method="one"):
    """Version of numpy.random.Generator that returns lnarrays

    Parameters
    ----------
    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
        The argument of `numpy.random.default_rng`. By default `None`.

    See Also
    --------
    numpy.random.Generator
    """
    obj: _pr.Generator

    def __init__(self, seed=None):
        super().__init__(_pr.default_rng(seed))

    def __getattr__(self, attr):
        if attr in {"bit_generator", "shuffle", "permutation"}:
            return getattr(self.obj, attr)
        return super().__getattr__(attr)


def default_rng(seed=None) -> LnGenerator:
    """Create a random number generator that returns lnarrays

    Parameters
    ----------
    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
        The argument of `numpy.random.default_rng`. By default `None`.

    Returns
    -------
    rng : LnGenerator
        A generator instance

    See Also
    --------
    numpy.random.default_rng
    """
    return LnGenerator(seed)


# =========================================================================
# Legacy RNG
# =========================================================================
beta = _wrap.one(_pr.beta)
binomial = _wrap.one(_pr.binomial)
bytes = _wrap.one(_pr.bytes)
chisquare = _wrap.one(_pr.chisquare)
choice = _wrap.one(_pr.choice)
dirichlet = _wrap.one(_pr.dirichlet)
exponential = _wrap.one(_pr.exponential)
f = _wrap.one(_pr.f)
gamma = _wrap.one(_pr.gamma)
geometric = _wrap.one(_pr.geometric)
get_state = _wrap.one(_pr.get_state)
gumbel = _wrap.one(_pr.gumbel)
hypergeometric = _wrap.one(_pr.hypergeometric)
laplace = _wrap.one(_pr.laplace)
logistic = _wrap.one(_pr.logistic)
lognormal = _wrap.one(_pr.lognormal)
logseries = _wrap.one(_pr.logseries)
multinomial = _wrap.one(_pr.multinomial)
multivariate_normal = _wrap.one(_pr.multivariate_normal)
negative_binomial = _wrap.one(_pr.negative_binomial)
noncentral_chisquare = _wrap.one(_pr.noncentral_chisquare)
noncentral_f = _wrap.one(_pr.noncentral_f)
normal = _wrap.one(_pr.normal)
pareto = _wrap.one(_pr.pareto)
permutation = _wrap.one(_pr.permutation)
poisson = _wrap.one(_pr.poisson)
power = _wrap.one(_pr.power)
rand = _wrap.one(_pr.rand)
randint = _wrap.one(_pr.randint)
randn = _wrap.one(_pr.randn)
random_integers = _wrap.one(_pr.random_integers)
random_sample = _wrap.one(_pr.random_sample)
rayleigh = _wrap.one(_pr.rayleigh)
seed = _wrap.one(_pr.seed)
set_state = _wrap.one(_pr.set_state)
shuffle = _wrap.one(_pr.shuffle)
standard_cauchy = _wrap.one(_pr.standard_cauchy)
standard_exponential = _wrap.one(_pr.standard_exponential)
standard_gamma = _wrap.one(_pr.standard_gamma)
standard_normal = _wrap.one(_pr.standard_normal)
standard_t = _wrap.one(_pr.standard_t)
triangular = _wrap.one(_pr.triangular)
uniform = _wrap.one(_pr.uniform)
vonmises = _wrap.one(_pr.vonmises)
wald = _wrap.one(_pr.wald)
weibull = _wrap.one(_pr.weibull)
zipf = _wrap.one(_pr.zipf)
ranf = _wrap.one(_pr.ranf)
random = _wrap.one(_pr.random)
sample = _wrap.one(_pr.sample)
