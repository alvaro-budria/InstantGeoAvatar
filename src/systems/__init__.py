systems = {}


def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator


def make(name, config, load_from_checkpoint=None, **kwargs):
    if load_from_checkpoint is None:
        system = systems[name](config, **kwargs)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config, **kwargs)
    return system


from . import InstantGeoAvatar, InstantGeoAvatar_preSDF