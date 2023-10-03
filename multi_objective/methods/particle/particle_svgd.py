import os
import sys
import numpy as np
import math
import random
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import model_from_dataset, calc_gradients, circle_points
from min_norm_solvers import MinNormSolver, gradient_normalizers
from ..base import BaseMethod

class MultipleModels(nn.Module):
    
    def __init__(self, n_nets, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList([model_from_dataset(method='particle', **kwargs) for _ in range(n_nets)])
        self.n_nets = n_nets
        
    def forward(self, batch):
        return [net(batch) for net in self.nets]
    
    def apply_noise(self, gamma):
        for net in self.nets:
            private_params = net.private_params() if hasattr(net, 'private_params') else []
            for name, param in net.named_parameters():
                not_private = all([p not in name for p in private_params])
                if not_private and param.requires_grad:
                    param.grad.data += torch.randn_like(param) * gamma
        
def get_kernel(rank, losses,  width = 1.): # losses.shape = (n_nets, n_tasks)
    n = losses.shape[0]
    pairwise_dist = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    h = pairwise_dist.quantile(0.5) / math.log(n)
    kernel_matrix = torch.exp( - pairwise_dist / ( width * h + 1e-8))
    kernel_matrix = kernel_matrix * (1 - torch.eye(n).to(rank))
    return kernel_matrix


class ParticleMethod(BaseMethod):

    def __init__(self, objectives, n_particles, normalization_type, width, **kwargs):
        self.objectives = objectives
        self.K = len(objectives)
        self.n_particles = n_particles
        self.normalization_type = normalization_type
        self.model = MultipleModels(n_particles, **kwargs)
        self.width = width
        
        self.rays = circle_points(n_particles, dim = self.K)
        
    def step(self, batch, rank):

        sol_per_net = []
        for net in self.model.module.nets:
            net = copy.deepcopy(net)
            grads, obj_values = calc_gradients(batch, net, self.objectives)
            
            gn = gradient_normalizers(grads, obj_values, self.normalization_type)
            for t in range(len(self.objectives)):
                for gr_i in grads[t]:
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            grads = [[v for v in grad.values()] for grad in grads]
            
            sol, _ = MinNormSolver.find_min_norm_element(grads)
            
            sol_per_net.append(sol)
            
        self.model.zero_grad()

        logits_per_net = self.model(batch)
        
        loss_all_nets = None
        loss_per_net = torch.zeros((self.n_particles, self.K)).to(rank)
        for i in range(self.n_particles):
            batch.update(logits_per_net[i])
            for (k, (a, objective)) in enumerate(zip(sol_per_net[i], self.objectives)):
                task_loss = objective(**batch)
                loss_per_net[i, k] = task_loss
            
        G = get_kernel(rank, loss_per_net, width = self.width)
        
        G_np = G.detach().cpu().numpy()
        Sol = np.stack(sol_per_net)
        Sol = G_np @ Sol / self.K
        
        for i in range(self.n_particles):
            loss = None
            batch.update(logits_per_net[i])
            for (k, (a, b, objective)) in enumerate(zip(Sol[i], self.rays[i], self.objectives)):
                w = 0.3 * a + 0.7 * b
                loss = w * loss_per_net[i, k] if not loss else loss + w * loss_per_net[i, k]
            loss_all_nets = loss if not loss_all_nets else loss_all_nets + loss
            
        loss_all_nets = loss_all_nets + 0.5 * G.sum()
        
        loss_all_nets.backward()
            
        return loss_all_nets.item(), 0
    
    def eval_step(self, batch):
        self.model.eval()
        return self.model(batch)

