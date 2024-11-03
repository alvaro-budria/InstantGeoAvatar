losses = {}


def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls
    return decorator


def make(name, config):
    return losses[name](config)


from . import InstantGeoAvatar