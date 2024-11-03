samplers = {}

def register(name):
    def decorator(cls):
        samplers[name] = cls
        return cls
    return decorator


def make(name, config):
    sampler = samplers[name](config)
    return sampler

from . import sampler