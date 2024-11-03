renderers = {}


def register(name):
    def decorator(cls):
        renderers[name] = cls
        return cls
    return decorator


def make(name, config):
    model = renderers[name](config)
    return model


from . import renderer_volsdf
