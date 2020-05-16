"""Wrapped version of module numpy.random
"""

import numpy.random as _pr
from . import wrappers as _wr
from ._lnarray import lnarray as _lnarray

__all__ = [
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

_wrap_one = _wr.make_wrap_one(_lnarray, "numpy_linalg.random")
beta = _wrap_one(_pr.beta)
binomial = _wrap_one(_pr.binomial)
bytes = _wrap_one(_pr.bytes)
chisquare = _wrap_one(_pr.chisquare)
choice = _wrap_one(_pr.choice)
dirichlet = _wrap_one(_pr.dirichlet)
exponential = _wrap_one(_pr.exponential)
f = _wrap_one(_pr.f)
gamma = _wrap_one(_pr.gamma)
geometric = _wrap_one(_pr.geometric)
get_state = _wrap_one(_pr.get_state)
gumbel = _wrap_one(_pr.gumbel)
hypergeometric = _wrap_one(_pr.hypergeometric)
laplace = _wrap_one(_pr.laplace)
logistic = _wrap_one(_pr.logistic)
lognormal = _wrap_one(_pr.lognormal)
logseries = _wrap_one(_pr.logseries)
multinomial = _wrap_one(_pr.multinomial)
multivariate_normal = _wrap_one(_pr.multivariate_normal)
negative_binomial = _wrap_one(_pr.negative_binomial)
noncentral_chisquare = _wrap_one(_pr.noncentral_chisquare)
noncentral_f = _wrap_one(_pr.noncentral_f)
normal = _wrap_one(_pr.normal)
pareto = _wrap_one(_pr.pareto)
permutation = _wrap_one(_pr.permutation)
poisson = _wrap_one(_pr.poisson)
power = _wrap_one(_pr.power)
rand = _wrap_one(_pr.rand)
randint = _wrap_one(_pr.randint)
randn = _wrap_one(_pr.randn)
random_integers = _wrap_one(_pr.random_integers)
random_sample = _wrap_one(_pr.random_sample)
rayleigh = _wrap_one(_pr.rayleigh)
seed = _wrap_one(_pr.seed)
set_state = _wrap_one(_pr.set_state)
shuffle = _wrap_one(_pr.shuffle)
standard_cauchy = _wrap_one(_pr.standard_cauchy)
standard_exponential = _wrap_one(_pr.standard_exponential)
standard_gamma = _wrap_one(_pr.standard_gamma)
standard_normal = _wrap_one(_pr.standard_normal)
standard_t = _wrap_one(_pr.standard_t)
triangular = _wrap_one(_pr.triangular)
uniform = _wrap_one(_pr.uniform)
vonmises = _wrap_one(_pr.vonmises)
wald = _wrap_one(_pr.wald)
weibull = _wrap_one(_pr.weibull)
zipf = _wrap_one(_pr.zipf)
ranf = _wrap_one(_pr.ranf)
random = _wrap_one(_pr.random)
sample = _wrap_one(_pr.sample)
