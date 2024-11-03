import math
import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn

from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info

from utils.misc import config_to_primitive, get_rank
from models.utils import get_activation
from systems.utils import update_module_step


def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]


def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        print('type(a):', type(a))
        print('type(b):', type(b))
        print('----')
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


class EmbedderHann(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        kick_in_iter = torch.tensor(self.kwargs["kick_in_epoch"],
                                    dtype=torch.float32)
        t = torch.clamp(self.kwargs['epoch_val'] - kick_in_iter, min=0.)
        N = self.kwargs["full_band_epoch"] - kick_in_iter
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx, 
                                                   min=0., max=1.))) / 2.
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder_Hann(multires, epoch_val, kick_in_epoch, full_band_epoch, is_identity=0, input_dims=3):
    if is_identity == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True, #False,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'periodic_fns' : [torch.sin, torch.cos],
                'epoch_val': epoch_val,
                'kick_in_epoch': kick_in_epoch,
                'full_band_epoch': full_band_epoch,
    }

    embedder_obj = EmbedderHann(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class FourierEncoderWithMLP(nn.Module):
    def __init__(self):
        super(FourierEncoderWithMLP, self).__init__()

        embed_fn, embed_dim = get_embedder_Hann(
            multires=10, epoch_val=-1,
            kick_in_epoch=-1, full_band_epoch=-1,
        )
        # MLP with 1 hidden layer of size 64 and SiLU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x, config):
        # get embedder function
        embed_fn, embed_dim = get_embedder_Hann(
            multires=config['multires'], epoch_val=config['epoch_val'],
            kick_in_epoch=config['kick_in_epoch'], full_band_epoch=config['full_band_epoch'],
        )
        # embed input
        x = embed_fn(x)
        # pass through MLP
        x = self.mlp(x)
        return x


class NonRigidMotionMLP_TCNN(nn.Module):
    def __init__(self,
                 pos_embed_size=3,
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 skips=None,
                 out_dim=3,
                 return_delta=False,
                 range_out=[-0.5, 0.5]):
        super(NonRigidMotionMLP_TCNN, self).__init__()

        config = {
            "otype": "FullyFusedMLP",  #CutlassMLP
            "activation": "ReLU",
            "output_activation": "ReLU",
            "n_neurons": 128,
            "n_hidden_layers": 4,
        }
        mlp_tcnn = get_mlp(
            n_input_dims=pos_embed_size+condition_code_size,
            n_output_dims=mlp_width,
            config=config,
        )
        res_mlp = nn.Linear(mlp_width+pos_embed_size, mlp_width)
        out_mlp = nn.Linear(mlp_width, out_dim)
        self.block_mlps = nn.ModuleList(
            [mlp_tcnn, res_mlp, nn.ReLU(), out_mlp]
        )

        self.return_delta = return_delta

        initseq(self.block_mlps[1:])

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros
        init_val = 1e-7
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, _embed, pos_xyz, condition_code, viewdirs=None, **_):
        h = torch.cat([condition_code, _embed], dim=-1)

        h = self.block_mlps[0](h)
        h = torch.cat([h, _embed], dim=-1)
        h = self.block_mlps[1](h)
        h = self.block_mlps[2](h)
        h = self.block_mlps[3](h)
        trans = h
        trans = 0.1 * torch.tanh(trans)

        if self.return_delta:
            return trans
        return pos_xyz + trans


# borrowed from HumaNeRF
class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 pos_embed_size=3,
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 skips=None,
                 out_dim=3,
                 return_delta=False,
                 range_out=[-0.5, 0.5]):
        super(NonRigidMotionMLP, self).__init__()
        self.range_out = range_out
        self.return_delta = return_delta
        self.skips = [4] if skips is None else skips
        
        block_mlps = [nn.Linear(pos_embed_size+condition_code_size,
                                mlp_width), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width+pos_embed_size, mlp_width), 
                               nn.ReLU()]
            else:
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, out_dim)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros
        init_val = 1e-7
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, _embed, pos_xyz, condition_code, viewdirs=None, **_):
        h = torch.cat([condition_code, _embed], dim=-1)
        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, _embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h
        trans = 0.1 * torch.tanh(trans)
        if self.return_delta:
            return trans
        return pos_xyz + trans


class VanillaFrequency(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get('n_masking_step', 0)
        self.update_step(None, None)  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq*x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (1. - torch.cos(math.pi * (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs)).clamp(0, 1))) / 2.
            rank_zero_debug(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config['otype'] = 'HashGrid'
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config['n_levels']
        self.n_features_per_level = config['n_features_per_level']
        self.start_level, self.start_step, self.update_steps = config['start_level'], config['start_step'], config['update_steps']
        self.current_level = self.start_level
        self.mask = torch.zeros(self.n_level * self.n_features_per_level, dtype=torch.float32, device=get_rank())

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step):
        current_level = min(self.start_level + max(global_step - self.start_step, 0) // self.update_steps, self.n_level)
        if current_level > self.current_level:
            rank_zero_info(f'Update grid level to {current_level}')
        self.current_level = current_level
        self.mask[:self.current_level * self.n_features_per_level] = 1.


class CompositeEncoding(nn.Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims

    def forward(self, x, *args):
        return self.encoding(x, *args) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)


class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False, device='cuda:0')
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings.to('cuda:0')

    def forward(self, t):
        return self.pos_embeddings[t, :]


def get_encoding(n_input_dims, config, xyz_scale: float, xyz_offset: float):
    # input suppose to be range [0, 1]
    if config.otype == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == 'ProgressiveBandHashGrid':
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding, include_xyz=config.get('include_xyz', False), xyz_scale=xyz_scale, xyz_offset=xyz_offset
    )
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.with_bias = config.get('with_bias', False)
        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False, with_bias=self.with_bias),
            self.make_activation()
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False, with_bias=self.with_bias),
                self.make_activation()
            ]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True, with_bias=self.with_bias)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])

    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, with_bias):
        layer = nn.Linear(dim_in, dim_out, bias=with_bias)
        if self.sphere_init:
            if is_last:
                if with_bias:
                    torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                if with_bias:
                    torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                if with_bias:
                    torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            if with_bias:
                torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)

def get_mlp(n_input_dims, n_output_dims, config):
    if config["otype"] == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            network = tcnn.Network(n_input_dims, n_output_dims, config_to_primitive(config))
    return network


class EncodingWithNetwork(nn.Module):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x, **kwargs):
        out = self.encoding(x)
        if "latent" in kwargs:
            cond = kwargs["latent"].reshape(-1, kwargs["latent"].shape[-1])
            if cond.shape[0] != out.shape[0]:
                cond = cond.expand(out.shape[0], cond.shape[-1])
            out = torch.cat([out, cond], dim=-1)
        return self.network(out)

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)


def get_encoding_with_network(
        n_input_dims, n_output_dims,
        encoding_config, network_config,
        xyz_scale: float=1., xyz_offset: float=0.,
        additional_input_dims: int=0
    ):
    # input suppose to be range [0, 1]
    if encoding_config.otype in ['VanillaFrequency', 'ProgressiveBandHashGrid'] \
        or network_config.otype in ['VanillaMLP']:
        encoding = get_encoding(n_input_dims, encoding_config, xyz_scale, xyz_offset)
        network = get_mlp(encoding.n_output_dims + additional_input_dims, n_output_dims, network_config)
        encoding_with_network = EncodingWithNetwork(encoding, network)
    else:
        with torch.cuda.device(get_rank()):
            encoding_with_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config_to_primitive(encoding_config),
                network_config=config_to_primitive(network_config)
            )
    return encoding_with_network