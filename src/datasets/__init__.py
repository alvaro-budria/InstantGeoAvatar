datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config, **kwargs):
    dataset = datasets[name](config, **kwargs)
    return dataset


from . import peoplesnapshot, SDF_SMPL, x_humans