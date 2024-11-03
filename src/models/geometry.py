import numpy as np

import torch
import torch.nn as nn

import models
from models.base import BaseModel
from models.utils import scale_anything, get_activation, cleanup, chunk_batch, LaplaceDensity
from models.network_utils import get_encoding_with_network
from systems.utils import update_module_step
from utils.misc import get_rank

EPS = 1e-1


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        threshold = 1e-3
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }

class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')
        self.radius = None
        self.vmin, self.vmax = None, None
        self.contraction_type = None  # assigned in system

    def forward_level(self, points, **kwargs):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax, **kwargs):
        def batch_func(x, **kwargs):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            # rv = self.forward_level(x).cpu()
            rv = self.forward_level(x, **kwargs).cpu()
            cleanup()
            return rv

        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices(), **kwargs)
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2])),
        ], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self, **kwargs):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_(self.vmin, self.vmax, **kwargs)
        try:
            vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        except IndexError:
            vmin, vmax = self.vmin, self.vmax
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(self.vmin.unsqueeze(0), self.vmax.unsqueeze(0)).squeeze()
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(self.vmin.unsqueeze(0), self.vmax.unsqueeze(0)).squeeze()
        mesh_fine = self.isosurface_(vmin_, vmax_, **kwargs)
        return mesh_fine


@models.register('volume-density')
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        assert self.config.xyz_encoding_config.otype == 'HashGrid'
        assert self.config.mlp_network_config.otype == 'FullyFusedMLP'
        self.encoding_with_network = get_encoding_with_network(
            self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config
        )
        self.register_buffer("center", torch.FloatTensor(self.config.center))
        self.register_buffer("scale", torch.FloatTensor(self.config.scale))

    def initialize(self, aabb):
        if hasattr(self, "aabb"):
            return
        c = (aabb[0] + aabb[1]) / 2
        s = (aabb[1] - aabb[0])
        self.center = c
        self.scale = s
        self.aabb = aabb
        self.radius = self.scale / 2.
        self.vmin, self.vmax = self.aabb[0].cpu(), self.aabb[1].cpu()

    def forward(self, points):
        # normalize pts to [0, 1]
        points = (points - self.center) / self.scale + 0.5
        assert points.min() >= -EPS and points.max() < 1 + EPS
        out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1], self.n_output_dims).float()
        density, feature = out[...,0], out
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        if 'feature_activation' in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature

    def forward_level(self, points):
        points = (points - self.center) / self.scale + 0.5
        assert points.min() >= -EPS and points.max() < 1 + EPS
        density = self.encoding_with_network(points.reshape(-1, self.n_input_dims)).reshape(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        return -density

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        # assert self.config.xyz_encoding_config.otype == 'HashGrid'
        assert self.config.mlp_network_config.otype == 'VanillaMLP'
        self.register_buffer("center", torch.FloatTensor(self.config.center))
        self.register_buffer("scale", torch.FloatTensor(self.config.scale))
        self.encoding_with_network = get_encoding_with_network(
            self.n_input_dims,
            self.n_output_dims,
            self.config.xyz_encoding_config,
            self.config.mlp_network_config,
            xyz_scale=self.scale.cuda(), xyz_offset=(self.center - 0.5 * self.scale).cuda(),
            # additional_input_dims=128,
            # additional_input_dims=3,  # HPE
        )
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 0.003)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == 'finite_difference':
            print(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")
        self.current_epoch = 0
        self.radius = torch.tensor([1.05]).cuda()

    def initialize(self, aabb, force_reinit=False):
        if hasattr(self, "aabb") and not force_reinit:
            return
        c = (aabb[0] + aabb[1]) / 2
        s = (aabb[1] - aabb[0])
        self.center = c
        self.scale = s
        self.encoding_with_network.encoding.xyz_scale = self.scale
        self.encoding_with_network.encoding.xyz_offset = self.center - 0.5 * self.scale
        self.aabb = aabb
        self.radius = self.scale / 2.
        self.vmin, self.vmax = self.aabb[0].cpu(), self.aabb[1].cpu()

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False, **kwargs):
        laplace = None
        # with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
        with torch.inference_mode(False):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic') or not self.training):
                if with_grad and self.grad_type == 'analytic' or True:
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                # normalize pts to [0, 1]
                points = (points - self.center) / self.scale + 0.5
                # assert points.min() >= -EPS and points.max() < 1 + EPS
                # clamp points
                points = points.clamp(0, 1)

                in_kwargs = dict()
                if "latent" in kwargs:
                    in_kwargs["latent"] = kwargs["latent"].view(-1, kwargs["latent"].shape[-1])
                if hasattr(self, "FourierWithMLP") and self.FourierWithMLP:
                    encoding_config = {
                        "multires": 10, "epoch_val": self.current_epoch, "kick_in_epoch": 0, "full_band_epoch": 10
                    }
                    encode_ = self.FourierWithMLP(points.view(-1, 3), encoding_config)
                    in_kwargs["latent"] = encode_.view(-1, encode_.shape[-1])
                out = self.encoding_with_network(points.view(-1, 3), **in_kwargs).view(*points.shape[:-1], self.n_output_dims).float()
                sdf, feature = out[...,0], out
                # if 'sdf_activation' in self.config:
                #     sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                # sdf = sdf - 0.5
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    # grad = torch.autograd.grad(
                    #     sdf, points_, grad_outputs=torch.ones_like(sdf),
                    #     create_graph=True, retain_graph=True, only_inputs=True
                    # )[0]
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + 0 * offsets)
                        points_d_ = (points_d_ - self.center) / self.scale + 0.5
                        points_d_ = points_d_ + offsets
                        points_d_ = points_d_.clamp(0, 1)
                        points_d_ = points_d_.view(-1, 3)
                        points_d_sdf = self.encoding_with_network(
                            points_d_, **in_kwargs
                        )
                        points_d_sdf = points_d_sdf[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps

                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)
        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            if laplace is None:
                laplace = torch.zeros_like(sdf)
            # assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points, **kwargs):
        if 'deformer' in kwargs:
            deformer = kwargs['deformer']
            points, valid, _ = deformer(points, eval_mode=True)

            get_sdf_density = kwargs['get_sdf_density']
            sdf, density_canonical, valid = get_sdf_density(points, valid)
            return sdf
        # raise ValueError('did you forget to implement forward_level?')

        # normalize pts to [0, 1]
        points = (points - self.center) / self.scale + 0.5
        assert points.min() >= -EPS and points.max() < 1 + EPS
        in_kwargs = dict()
        if "latent" in kwargs:
            in_kwargs["latent"] = kwargs["latent"].view(-1, kwargs["latent"].shape[-1])
        if hasattr(self, "FourierWithMLP") and self.FourierWithMLP:
            encoding_config = {
                "multires": 10, "epoch_val": self.current_epoch, "kick_in_epoch": 0, "full_band_epoch": 10
            }
            encode_ = self.FourierWithMLP(points.view(-1, 3), encoding_config)
            in_kwargs["latent"] = encode_.view(-1, encode_.shape[-1])
        sdf = self.encoding_with_network(points.view(-1, 3), **in_kwargs).view(*points.shape[:-1], self.n_output_dims).float()[...,0]
        # sdf = self.encoding_with_network(points.view(-1, 3)).view(*points.shape[:-1], self.n_output_dims).float()[...,0]
        # if 'sdf_activation' in self.config:
        #     sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        # sdf = sdf - 0.5
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)
    # def update_step(self, epoch, global_step):
    #     update_module_step(self.encoding, epoch, global_step)
    #     update_module_step(self.network, epoch, global_step)
        if self.grad_type == 'finite_difference':
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                # grid_size = 2 * self.config.radius / grid_res
                grid_size = 2 * self.radius.max().item() / grid_res if self.radius is not None else 1.05 / grid_res
                if grid_size != self._finite_difference_eps:
                    print(f"Update finite_difference_eps to {grid_size}")
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")

@models.register('volume-volsdf')
class VolumeVolSDF(VolumeSDF):
    def __init__(self, config):
        super().__init__(config)
        self.density = LaplaceDensity(**config.density)

    def get_beta(self):
        return self.density.get_beta()

    def get_density(self, sdfs, beta=None):
        return self.density.density_func(sdfs, beta=beta)
