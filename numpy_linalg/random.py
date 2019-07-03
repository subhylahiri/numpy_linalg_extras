"""Wrapped version of module numpy.random
"""

import numpy.random as _pr
from . import _ln_wrap as _wr

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

beta = _wr.wrap_one(_pr.beta)
binomial = _wr.wrap_one(_pr.binomial)
bytes = _wr.wrap_one(_pr.bytes)
chisquare = _wr.wrap_one(_pr.chisquare)
choice = _wr.wrap_one(_pr.choice)
dirichlet = _wr.wrap_one(_pr.dirichlet)
exponential = _wr.wrap_one(_pr.exponential)
f = _wr.wrap_one(_pr.f)
gamma = _wr.wrap_one(_pr.gamma)
geometric = _wr.wrap_one(_pr.geometric)
get_state = _wr.wrap_one(_pr.get_state)
gumbel = _wr.wrap_one(_pr.gumbel)
hypergeometric = _wr.wrap_one(_pr.hypergeometric)
laplace = _wr.wrap_one(_pr.laplace)
logistic = _wr.wrap_one(_pr.logistic)
lognormal = _wr.wrap_one(_pr.lognormal)
logseries = _wr.wrap_one(_pr.logseries)
multinomial = _wr.wrap_one(_pr.multinomial)
multivariate_normal = _wr.wrap_one(_pr.multivariate_normal)
negative_binomial = _wr.wrap_one(_pr.negative_binomial)
noncentral_chisquare = _wr.wrap_one(_pr.noncentral_chisquare)
noncentral_f = _wr.wrap_one(_pr.noncentral_f)
normal = _wr.wrap_one(_pr.normal)
pareto = _wr.wrap_one(_pr.pareto)
permutation = _wr.wrap_one(_pr.permutation)
poisson = _wr.wrap_one(_pr.poisson)
power = _wr.wrap_one(_pr.power)
rand = _wr.wrap_one(_pr.rand)
randint = _wr.wrap_one(_pr.randint)
randn = _wr.wrap_one(_pr.randn)
random_integers = _wr.wrap_one(_pr.random_integers)
random_sample = _wr.wrap_one(_pr.random_sample)
rayleigh = _wr.wrap_one(_pr.rayleigh)
seed = _wr.wrap_one(_pr.seed)
set_state = _wr.wrap_one(_pr.set_state)
shuffle = _wr.wrap_one(_pr.shuffle)
standard_cauchy = _wr.wrap_one(_pr.standard_cauchy)
standard_exponential = _wr.wrap_one(_pr.standard_exponential)
standard_gamma = _wr.wrap_one(_pr.standard_gamma)
standard_normal = _wr.wrap_one(_pr.standard_normal)
standard_t = _wr.wrap_one(_pr.standard_t)
triangular = _wr.wrap_one(_pr.triangular)
uniform = _wr.wrap_one(_pr.uniform)
vonmises = _wr.wrap_one(_pr.vonmises)
wald = _wr.wrap_one(_pr.wald)
weibull = _wr.wrap_one(_pr.weibull)
zipf = _wr.wrap_one(_pr.zipf)
ranf = _wr.wrap_one(_pr.ranf)
random = _wr.wrap_one(_pr.random)
sample = _wr.wrap_one(_pr.sample)
