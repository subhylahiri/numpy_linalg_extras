# -*- coding: utf-8 -*-
"""Tools for dealing with version numbering
"""
import packaging.version as _pkv


def _decode(str_like):
    """Convert to string from bytes/str/...
    """
    if isinstance(str_like, bytes):
        return str(str_like, 'utf-8')
    return str(str_like)


def get_version(arg):
    """Get version from module/string

    Given a module, version string or `packaging.version.Version` object
    it returns the corrsponding `Version` object. For non-`str`/`Version`
    arguments, those without a `__version__` attribute return `Version('0')`.
    """
    if isinstance(arg, (_pkv.Version, _pkv.LegacyVersion)):
        return arg
    if isinstance(arg, (str, bytes)):
        return _pkv.parse(_decode(arg))
    try:
        return _pkv.parse(_decode(arg.__version__))
    except AttributeError:
        pass
    return _pkv.parse('0')


def max_version(*modules):
    """Maximum version number of a sequence of modules/version strings

    See `get_version` for how version numbers are extracted. They are compared
    as `packaging.version.Version` objects.
    """
    return str(max(get_version(x) for x in modules))


def min_version(*modules):
    """Minimum version number of a sequence of modules/version strings

    See `get_version` for how version numbers are extracted. They are compared
    as `packaging.version.Version` objects.
    """
    return str(min(get_version(x) for x in modules))
