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


class LnGenerator(_wr.WrappedClass, wrappers=_wrap, method="check"):
    """Version of numpy.random.Generator that returns lnarrays

    Parameters
    ----------
    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
        The argument of `numpy.random.default_rng`. By default `None`.

    See Also
    --------
    numpy.random.Generator
    """
    _obj: _pr.Generator

    def __init__(self, seed=None):
        super().__init__(_pr.default_rng(seed))

    def __getattr__(self, attr):
        if attr in {"bit_generator", "shuffle", "bytes"}:
            return getattr(self._obj, attr)
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
beta = _wrap.check(_pr.beta)
binomial = _wrap.check(_pr.binomial)
bytes = _wrap.check(_pr.bytes)
chisquare = _wrap.check(_pr.chisquare)
choice = _wrap.check(_pr.choice)
dirichlet = _wrap.check(_pr.dirichlet)
exponential = _wrap.check(_pr.exponential)
f = _wrap.check(_pr.f)
gamma = _wrap.check(_pr.gamma)
geometric = _wrap.check(_pr.geometric)
get_state = _wrap.check(_pr.get_state)
gumbel = _wrap.check(_pr.gumbel)
hypergeometric = _wrap.check(_pr.hypergeometric)
laplace = _wrap.check(_pr.laplace)
logistic = _wrap.check(_pr.logistic)
lognormal = _wrap.check(_pr.lognormal)
logseries = _wrap.check(_pr.logseries)
multinomial = _wrap.check(_pr.multinomial)
multivariate_normal = _wrap.check(_pr.multivariate_normal)
negative_binomial = _wrap.check(_pr.negative_binomial)
noncentral_chisquare = _wrap.check(_pr.noncentral_chisquare)
noncentral_f = _wrap.check(_pr.noncentral_f)
normal = _wrap.check(_pr.normal)
pareto = _wrap.check(_pr.pareto)
permutation = _wrap.check(_pr.permutation)
poisson = _wrap.check(_pr.poisson)
power = _wrap.check(_pr.power)
rand = _wrap.check(_pr.rand)
randint = _wrap.check(_pr.randint)
randn = _wrap.check(_pr.randn)
random_integers = _wrap.check(_pr.random_integers)
random_sample = _wrap.check(_pr.random_sample)
rayleigh = _wrap.check(_pr.rayleigh)
seed = _wrap.check(_pr.seed)
set_state = _wrap.check(_pr.set_state)
shuffle = _wrap.check(_pr.shuffle)
standard_cauchy = _wrap.check(_pr.standard_cauchy)
standard_exponential = _wrap.check(_pr.standard_exponential)
standard_gamma = _wrap.check(_pr.standard_gamma)
standard_normal = _wrap.check(_pr.standard_normal)
standard_t = _wrap.check(_pr.standard_t)
triangular = _wrap.check(_pr.triangular)
uniform = _wrap.check(_pr.uniform)
vonmises = _wrap.check(_pr.vonmises)
wald = _wrap.check(_pr.wald)
weibull = _wrap.check(_pr.weibull)
zipf = _wrap.check(_pr.zipf)
ranf = _wrap.check(_pr.ranf)
random = _wrap.check(_pr.random)
sample = _wrap.check(_pr.sample)
