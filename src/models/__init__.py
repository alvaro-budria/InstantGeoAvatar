models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model


from . import InstantGeoAvatar, geometry, texture, snarf_deformer, SMPL_embedding, occupancy_grid
