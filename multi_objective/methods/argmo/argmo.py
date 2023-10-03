import torch
import torch.nn as nn
import numpy as np

from utils import num_parameters, model_from_dataset, circle_points, kernel_functional_rbf
from ..base import BaseMethod
from .hv_maximization import HvMaximization

class ARGMOMethod(BaseMethod):

    def __init__(self, objectives, alpha, lamda, dim, n_test_rays, n_particles, reference_point, explicit, **kwargs):
        """
        Instanciate the cosmos solver.

        Args:
            objectives: A list of objectives
            alpha: Dirichlet sampling parameter (list or float)
            lamda: Cosine similarity penalty
            dim: Dimensions of the data
            n_test_rays: The number of test rays used for evaluation.
        """
        self.objectives = objectives
        self.K = len(objectives)
        self.alpha = alpha
        self.n_test_rays = n_test_rays
        self.lamda = lamda
        self.explicit = explicit
        
        self.n_particles = n_particles
        self.hv_weights = HvMaximization(n_mo_sol=n_particles, n_mo_obj=len(objectives), ref_point=reference_point, obj_space_normalize=True)
        dim = list(dim)

        if explicit:
            dim[0] = self.K
            self.model = model_from_dataset(method='argmo', condition=True, dim=dim, n_objectives=len(objectives), explicit = explicit, **kwargs).cuda()
        else:
            self.model = model_from_dataset(method='argmo', condition=True, dim=dim, n_objectives=len(objectives), **kwargs).cuda()
        self.particles = (torch.rand((n_particles, self.K-1))*np.pi/2).cuda()
        self.particles.requires_grad = True

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))


    def step(self, batch):
        loss_total = None
        rand_matrix = (0.5*np.pi*torch.rand_like(self.particles)).cuda()
        self.particles.data = torch.clamp(self.particles.data, min=1e-6, max=np.pi/2-1e-6)
        self.particles.data = torch.where(self.particles.data < 1e-5, rand_matrix.detach(), self.particles.data)
        self.particles.data = torch.where(self.particles.data > np.pi/2-1e-5, rand_matrix.detach(), self.particles.data)
        
        if batch['stage'] == 0:
            # this is basically cosmos for particle #pi
            pi = batch['pi']
            if batch['use_p']:
                batch['alpha'] = self.convert_polar(self.particles[pi])
            else:
                if isinstance(self.alpha, list):
                    batch['alpha'] = torch.from_numpy(np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()).cuda()
                else:
                    batch['alpha'] = torch.from_numpy(np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()).cuda()

            self.model.zero_grad()
            logits = self.model(batch)
            batch.update(logits)
            task_losses = []
            for a, objective in zip(batch['alpha'], self.objectives):
                task_loss = objective(**batch)
                loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
                task_losses.append(task_loss)
            cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses), batch['alpha'], dim=0)
            loss_total -= self.lamda * cossim
            # print("task_losses: ", [t.item() for t in task_losses], "alpha: ", batch['alpha'].data.cpu().numpy())
            loss_total.backward()
        elif batch['stage'] == 1:
            # optimize the particles wrt hv
            total_task_losses = []
            for i in range(self.n_particles):
                batch['alpha'] = self.convert_polar(self.particles[i])
                self.model.zero_grad()
                logits = self.model(batch)
                batch.update(logits)
                task_losses = []
                for a, objective in zip(batch['alpha'], self.objectives):
                    task_loss = objective(**batch)
                    if self.particles.grad is not None:
                        self.particles.grad.zero_()
                    task_losses.append(task_loss.unsqueeze(0))
                total_task_losses.append(torch.cat(task_losses))

            total_losses = torch.stack(total_task_losses)

            if batch['method'] == 'hv':
                weights = (self.hv_weights.compute_weights(total_losses.cpu().detach().numpy().T)).T.cuda()
                loss_total = torch.sum(total_losses*weights*100)
                loss_total.backward()
            elif batch['method'] == 'kernel':
                kernel_matrix = kernel_functional_rbf(total_losses, h=batch['const'])
                kernel_grad = - 0.5 * torch.autograd.grad(kernel_matrix.sum(), self.particles, allow_unused=True)[0]
                self.particles.grad = - kernel_grad / self.n_particles

            loss_total = torch.sum(total_losses)


        return loss_total.item()


    def eval_step(self, batch, test_rays=None):
        self.model.eval()
        logits = []
        with torch.no_grad():
            if test_rays is None:
                test_rays = self.particles.detach()

            for ray in test_rays:
                batch['alpha'] = self.convert_polar(ray)
                logits.append(self.model(batch))
        return logits

    def convert_polar(self, x):
        f = []
        for i in range(0, self.K):
            if i == 0:
                f.append(torch.prod(torch.sin(x[:x.shape[0] - i])).unsqueeze(0))
            else:
                f.append(torch.prod(torch.sin(x[:x.shape[0] - i])) * torch.cos(x[x.shape[0] - i]).unsqueeze(0))
        return torch.cat(f)

