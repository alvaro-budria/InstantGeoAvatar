class BaseSampler():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup()

    def sample(mask, *args):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def update_step(self, epoch, global_step):
        pass

    def train(self, mode=True):
        return super().train(mode=mode)

    def eval(self):
        return super().eval()