import torch
import torch.nn.functional as F

import models
from models.base import BaseModel


@models.register('occupancy-grid')
class OccupancyGrid(BaseModel):
    """Occupancy grid. Allows for empty space skipping during ray marching."""
    def setup(self) -> None:
        self.grid_size = self.config.grid_size
        idx = torch.arange(0, self.grid_size)
        coords = torch.meshgrid((idx, idx, idx), indexing="ij")
        coords = torch.stack(coords, dim=-1)
        coords = coords.reshape(-1, 3) / self.grid_size
        self.coords = coords.cuda()

        self.register_buffer("density_cached", torch.zeros_like(self.coords[:, 0]))
        self.register_buffer("occupancy_field", torch.zeros(self.grid_size, self.grid_size, self.grid_size, dtype=torch.bool))
        self.scene_aabb = None
        self.active = True
        # when animating with poses outside of the training distribution, the occupancy grid prevents
        # the renderer from rendering outside of the occupancy grid, missing parts of the animated avatar,
        # so we disable it in that case

    @property
    def min_corner(self):
        return self.scene_aabb[0]

    @property
    def max_corner(self):
        return self.scene_aabb[1]

    def update(self, deformer, sigma_fn, step):
        if self.scene_aabb is None:
            bbox = deformer.get_bbox_deformed()
            self.scene_aabb = bbox

        coords = (
            self.coords + torch.rand_like(self.coords) / self.grid_size
        ) * (self.scene_aabb[1] - self.scene_aabb[0]) + self.scene_aabb[0]
        with torch.enable_grad():
            density, _ = sigma_fn(coords, eval_mode=False)
        density = density.clip(min=0)

        old_occupancy_field = self.occupancy_field.reshape(-1)
        self.density_cached = torch.maximum(self.density_cached * 0.8, density.detach())

        self.occupancy_field = 1 - torch.exp(0.01 * -self.density_cached)
        self.occupancy_field = self.occupancy_field.reshape(self.grid_size, self.grid_size, self.grid_size)
        self.occupancy_field = F.max_pool3d(self.occupancy_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        self.occupancy_field = self.occupancy_field > torch.clamp(self.occupancy_field.mean(), max=0.01)

        if step > 1000:
            occupied = old_occupancy_field
        else:
            occupied = self.occupancy_field.reshape(-1)
        return coords, density, occupied

    @torch.no_grad()
    def initialize(self, deformer, sigma_fn, n_it=5):
        self.scene_aabb = deformer.get_bbox_deformed()

        density = torch.zeros_like(self.coords[..., 0])
        for _ in range(n_it):
            delta_coords = torch.rand_like(self.coords)
            coords = (self.coords + delta_coords / self.grid_size) * (self.scene_aabb[1] - self.scene_aabb[0]) + self.scene_aabb[0]
            d, _ = sigma_fn(coords, eval_mode=not self.training)
            density = torch.maximum(density, d)

        self.occupancy_field = 1 - torch.exp(0.01 * -density)

        self.occupancy_field = self.occupancy_field.reshape(self.grid_size, self.grid_size, self.grid_size)
        self.occupancy_field = F.max_pool3d(self.occupancy_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        self.occupancy_field = self.occupancy_field > 0.00


@models.register('occupancy-grid-alpha')
class OccupancyGrid(BaseModel):
    """Occupancy grid. Allows for empty space skipping during ray marching."""
    def setup(self) -> None:
        self.grid_size = self.config.grid_size
        idx = torch.arange(0, self.grid_size)
        coords = torch.meshgrid((idx, idx, idx), indexing="ij")
        coords = torch.stack(coords, dim=-1)
        coords = coords.reshape(-1, 3) / self.grid_size
        self.coords = coords.cuda()

        self.register_buffer("occupancy_field", torch.zeros(self.grid_size, self.grid_size, self.grid_size, dtype=torch.bool))
        self.scene_aabb = None

    @property
    def min_corner(self):
        return self.scene_aabb[0]

    @property
    def max_corner(self):
        return self.scene_aabb[1]

    def update(self, deformer, alpha_fn, step):
        if self.scene_aabb is None:
            bbox = deformer.get_bbox_deformed()
            self.scene_aabb = bbox

        coords = (
            self.coords + torch.rand_like(self.coords) / self.grid_size
        ) * (self.scene_aabb[1] - self.scene_aabb[0]) + self.scene_aabb[0]
        with torch.enable_grad():
            alpha, _ = alpha_fn(coords)
        alpha = alpha.clip(min=0)

        old_occupancy_field = self.occupancy_field.reshape(-1)

        self.occupancy_field = torch.maximum(old_occupancy_field * 0.8, alpha.detach())
        self.occupancy_field = self.occupancy_field.reshape(self.grid_size, self.grid_size, self.grid_size)
        self.occupancy_field = F.max_pool3d(self.occupancy_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        self.occupancy_field = self.occupancy_field > torch.clamp(self.occupancy_field.mean(), max=0.01)

        if step > 1000:
            occupied = old_occupancy_field
        else:
            occupied = self.occupancy_field.reshape(-1)
        return coords, alpha, occupied

    @torch.no_grad()
    def initialize(self, deformer, alpha_eval_fn, n_it=5):
        self.scene_aabb = deformer.get_bbox_deformed()

        alpha = torch.zeros_like(self.coords[..., 0])
        for i in range(n_it):
            delta_coords = torch.rand_like(self.coords)
            coords = (self.coords + delta_coords / self.grid_size) * (self.scene_aabb[1] - self.scene_aabb[0]) + self.scene_aabb[0]
            d, _ = alpha_eval_fn(coords)
            alpha = torch.maximum(alpha, d)

        self.occupancy_field = alpha
        self.occupancy_field = self.occupancy_field.reshape(self.grid_size, self.grid_size, self.grid_size)
        self.occupancy_field = F.max_pool3d(self.occupancy_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        self.occupancy_field = self.occupancy_field > 0.00