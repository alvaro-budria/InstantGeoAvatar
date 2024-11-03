import numpy as np
from typing import Union

import pytorch3d
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

import torch
import torch.nn.functional as F

import losses
import models
import systems
from systems.base import BaseSystem
from systems.utils import parse_scheduler, get_parameters


@systems.register('instantgeoavatar-system-preSDF')
class InstantAvatarSystem(BaseSystem):
    def prepare(self, datamodule):
        self.loss_fn = losses.make('base-loss', self.config.system.loss)  # placeholder to avoid crash
        self.criteria = {
            'mape': MAPE,
        }
        self.datamodule = datamodule
        self.train_num_samples = 2 << 16
        self.val_num_samples = 2 << 15
        self.lambda_eikonal_sdf = 0.01
        self.lambda_mape = 1.

        self.model = models.make(self.config.model.name, self.config.model)

    def configure_optimizers(self):
        params = list()
        # geometry encoding
        params += [{'params': get_parameters(self.model.geometry.encoding_with_network, 'encoding'), 'name': 'geometry_encoding', **self.config.system.optimizer.params.geometry_encoding}]
        # geometry network
        params += [{'params': get_parameters(self.model.geometry.encoding_with_network, 'network'), 'name': 'geometry_network', **self.config.system.optimizer.params.geometry_network}]
        print('optimizer params:', params)
        optim = getattr(torch.optim, self.config.system.optimizer.name)(params, **self.config.system.optimizer.args)

        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })

        if hasattr(self.model, 'density'):
            self.model.geometry.density = self.model.density

        return ret

    def forward(self, batch):
        smpl_pose = batch['body_pose'] * 0.
        points_canonical = batch['pts']
        points_canonical.requires_grad = True
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                sdf, grad = self.model.sdf_fn_canonical(points_canonical)
        return [sdf, grad]

    def preprocess_data(self, batch, stage):
        """ Online sampling
        """
        aabb = self.model.geometry.aabb.cpu().numpy() if hasattr(self.model.geometry, 'aabb') else np.array([[-1,-1,-1], [1.,1.,1.]]) * 0.4
        points, sdfs, n = self.dataset.resample(self.train_num_samples, aabb)
        sdfs = torch.from_numpy(sdfs).cuda().squeeze().float()
        points = torch.from_numpy(points).cuda().float()
        n = None
        batch.update({
            'd': sdfs,
            'pts': points,
            'n': n,
        })

    def training_step(self, batch, *args, **kwargs):
        # Since here we learn the canonical SDF Only,
        # the deformer is not used so we do not need to use it, but we still initialize it
        # because we need to initialize the geometry.
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                self.model.deformer.prepare_deformer(batch)
                self.model.deformer.initialized = False  # keep it uninitialized, so that when the training of the SDF is done, we can initialize it with the SMPL params

        # we still need to initialize the geometry to fit the scene inside the volume
        self.model.geometry.initialize(self.model.deformer.scene_aabb)

        losses = {}
        losses["loss"] = 0.

        out = self(batch)
        sdf, grad = out

        loss_eikonal = ((torch.linalg.norm(grad, ord=2, dim=-1) - 1.)**2).mean()
        losses["loss_eikonal"] = loss_eikonal
        self.log('train/loss_eikonal', loss_eikonal)
        losses["loss"] += loss_eikonal * self.C(self.lambda_eikonal_sdf, self.global_step, self.current_epoch)

        loss_mape = self.criteria['mape'](sdf, batch['d'])
        losses["loss_mape"] = loss_mape
        self.log('train/loss_mape', loss_mape)
        losses["loss"] += loss_mape * self.C(self.lambda_mape, self.global_step, self.current_epoch)

        # add each and all params of the model multiplied by 0
        losses["loss"] += sum(0 * p.sum() for p in self.model.parameters())

        for k, v in losses.items():
            self.log(f"train/{k}", v)

        if 'inv_s' in out:
            self.log('train/inv_s', out['inv_s'], prog_bar=True)
        elif hasattr(self.model.geometry, 'density'):
            self.log('train/beta', self.model.geometry.density.beta, prog_bar=True)

        if self.precision == 16:
            self.log("precision/scale", self.trainer.precision_plugin.scaler.get_scale())

        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                self.model.deformer.prepare_deformer(batch)
                self.model.deformer.initialized = False
                # keep it uninitialized, so that when the training of the SDF is done,
                # we can initialize it with the SMPL params

        out = self(batch)
        sdf, grad = out
        eikonal = ((torch.linalg.norm(grad, ord=2, dim=-1) - 1.)**2).mean()
        MAPE = self.criteria['mape'](sdf, batch['d'])
        return {
            'mape': MAPE,
            'eikonal': eikonal,
            'index': batch['index'],
        }

    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                for oi, index in enumerate(step_out['index']):
                    out_set[index[0].item()] = {
                        'eikonal': step_out['eikonal'][oi],
                        'mape': step_out['mape'][oi],
                    }
            loss_eikonal = torch.mean(torch.stack([o['eikonal'] for o in out_set.values()]))
            loss_mape = torch.mean(torch.stack([o['mape'] for o in out_set.values()]))
            self.log('val/loss_eikonal', loss_eikonal, prog_bar=True, rank_zero_only=True)
            self.log('val/loss_mape', loss_mape, prog_bar=True, rank_zero_only=True)

        mesh_pred = self.export('val', return_mesh=True)
        mesh_pred = self.assemble_mesh(**mesh_pred, extract_large_component=False)
        mesh_gt = self.dataset.mesh_gt
        # compute Chamfer L2 distance
        _deformed = torch.from_numpy(mesh_pred.vertices).double().cuda().unsqueeze(0)
        _gt = torch.from_numpy(mesh_gt.vertices).double().cuda()[None,...]
        chamfer = chamfer_distance(_deformed * 100, _gt * 100)  # in cm  # * 1000  # in mm
        print('Step: {}, Chamfer L2 distance: {}'.format(self.global_step, chamfer[0].item()))
        # compute normal consistency
        try:
            normal_cons = normal_consistency_vertex(mesh_pred, mesh_gt)
        except Exception as e:
            normal_cons = 0.
        print('Step: {}, Normal Consistency: {}'.format(self.global_step, normal_cons))

    def export(self, split, return_mesh=False):
        self.model.geometry.initialize(self.model.deformer.scene_aabb)
        if self.model.geometry.config.name == 'volume-volsdf-avatar':
            mesh = self.model.export(self.config.export, self.model.deformer.smpl_params['body_pose'])
        else:
            mesh = self.model.export(self.config.export)
        try:
            self.save_mesh(
                f"{split}/it{self.global_step}-{self.config.model.geometry.isosurface.resolution}.obj",
                **mesh,
                extract_large_component=True
            )
        except Exception as e:
            self.print("Error saving mesh: ", e)
        if return_mesh:
            return mesh


def normal_consistency_vertex(pred_trimesh, gt_trimesh):
    """
    :param pred: predicted trimesh
    :param gt trimesh: GT mesh trimesh
    """
    pred_vertices = np.array(pred_trimesh.vertices)
    pred_normals = np.array(pred_trimesh.vertex_normals)

    gt_vertices = np.array(gt_trimesh.vertices)
    gt_normals = np.array(gt_trimesh.vertex_normals)

    pred_verts_torch = torch.from_numpy(pred_vertices).double().unsqueeze(0).cuda()
    gt_verts_torch = torch.from_numpy(gt_vertices).double().unsqueeze(0).cuda()

    knn_ret = pytorch3d.ops.knn_points(gt_verts_torch, pred_verts_torch)
    p_idx = knn_ret.idx.squeeze(-1).detach().cpu().numpy()

    pred_normals = pred_normals[p_idx, :]

    consistency = 1 - np.linalg.norm(pred_normals - gt_normals, axis=-1).mean()

    return consistency

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights,
    batch_reduction: Union[str, None],
    point_reduction: str,
    norm: int,
    abs_cosine: bool,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
        cham_norm_x = cham_norm_x.sum(1)  # (N,)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        if return_normals:
            cham_norm_x /= x_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div
            if return_normals:
                cham_norm_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals

def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):

    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        return cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_normals,
            x_normals,
            weights,
            batch_reduction,
            point_reduction,
            norm,
            abs_cosine,
        )
        return (
            cham_x + cham_y,
            (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
        )


def MAPE(pred, target, reduction='mean'):
    # pred, target: [B, 1], torch tenspr
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    if reduction == 'mean':
        loss = loss.mean()
    
    return loss