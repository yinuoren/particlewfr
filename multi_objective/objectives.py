import torch
import autograd
import math


def from_name(names, task_names, mtl, **kwargs):
    objectives = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
        'L1Regularization': L1Regularization,
        'L2Regularization': L2Regularization,
        'ddp': DDPHyperbolicTangentRelaxation,
        'deo': DEOHyperbolicTangentRelaxation,
        'ListNetLoss': ListNetLoss,
        'InnerProductUtility': InnerProductUtility,
        'Fonseca1': Fonseca1,
        'Fonseca2': Fonseca2,
        'ZDT3_1': ZDT3_1,
        'ZDT3_2': ZDT3_2,
    }


    if task_names is not None:
        if mtl:
            return [objectives[n](label_name = "labels_{}".format(t), logits_name = "logits_{}".format(t)) for n, t in zip(names, task_names)]
        else:
            return [objectives[n](label_name = "labels_{}".format(t)) for n, t in zip(names, task_names)]
    else:
        return [objectives[n]() for n in names]
    
    
class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    
    def __init__(self, label_name='labels', logits_name='logits'):
        super().__init__(reduction='mean')
        self.label_name = label_name
        self.logits_name = logits_name
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        return super().__call__(logits, labels)


class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):
    
    def __init__(self, label_name='labels', logits_name='logits', pos_weight=None):
        super().__init__(reduction='mean', pos_weight=torch.Tensor([pos_weight]).cuda() if pos_weight else None)
        self.label_name = label_name
        self.logits_name = logits_name
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if labels.dtype != torch.float:
            labels = labels.float()
        return super().__call__(logits, labels)


class MSELoss(torch.nn.MSELoss):

    def __init__(self, label_name='labels', logits_name='logits'):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        return super().__call__(logits, labels)


class ListNetLoss():
    def __init__(self, label_name='labels', logits_name='logits'):
        self.label_name = label_name
        self.logits_name = logits_name
        
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        n = kwargs['n']
        if logits.ndim == 2:
            logits = logits.unsqueeze(-1)
        if labels.ndim == 2:
            labels = labels.unsqueeze(-1)
            
        mask = torch.arange(logits.shape[1], device = labels.device)[None, :, None] < n[:, None, None]
        
        masked_logits = torch.where(mask, logits, float('-inf'))
        masked_labels = torch.where(mask, labels, float('-inf'))
        
        logits_smax = torch.nn.functional.softmax(masked_logits, dim=1)
        labels_smax = torch.nn.functional.softmax(masked_labels, dim=1)
        
        return - torch.sum(labels_smax * torch.log(logits_smax + 1e-10), dim=1).mean()


class InnerProductUtility():
    def __init__(self, label_name='labels', logits_name='logits'):
        self.label_name = label_name
        self.logits_name = logits_name
        
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        n = kwargs['n']
        if logits.ndim == 2:
            logits = logits.unsqueeze(-1)
        if labels.ndim == 2:
            labels = labels.unsqueeze(-1)
            
        mask = torch.arange(logits.shape[1], device = labels.device)[None, :, None] < n[:, None, None]
        
        masked_logits = torch.where(mask, logits, float('-inf'))
        logits_smax = torch.nn.functional.softmax(masked_logits, dim=1)
        
        return - torch.sum(labels * logits_smax, dim=1).mean()


class L1Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1)


class L2Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=2)


class DDPHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[sensible_attribute.bool()]
        s_positive = logits[~sensible_attribute.bool()]

        return 1/n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))))


class DEOHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
        s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

        return 1/n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))))



"""
Popular problem proposed by

    Carlos Manuel Mira da Fonseca. Multiobjective genetic algorithms with 
    application to control engineering problems. PhD thesis, University of Sheffield, 1995.

with a concave pareto front.

$ \mathcal{L}_1(\theta) = 1 - \exp{ - || \theta - 1 / \sqrt{d} || ^ 2 ) $
$ \mathcal{L}_1(\theta) = 1 - \exp{ - || \theta + 1 / \sqrt{d} || ^ 2 ) $

with $\theta \in R^d$ and $ d = 100$
"""

class Fonseca1():
    
    def __init__(self, label_name='labels', logits_name='logits'):
        self.label_name = label_name
        self.logits_name = logits_name

    @staticmethod
    def f1(theta):
        sum1 = torch.sum((theta - 1.0 / math.sqrt(theta.numel())) ** 2)
        f1 = 1 - torch.exp(-sum1)
        return f1
    
    def __call__(self, logits, **kwargs):
        return Fonseca1.f1(logits)
    
    
class Fonseca2():
    
    def __init__(self, label_name='labels', logits_name='logits'):
        self.label_name = label_name
        self.logits_name = logits_name

    @staticmethod
    def f2(theta):
        sum2 = torch.sum((theta + 1.0 / math.sqrt(theta.numel())) ** 2)
        f2 = 1 - torch.exp(-sum2)
        return f2
    
    def __call__(self, logits, **kwargs):
        return Fonseca2.f2(logits)


class ZDT3_1():
    
    def __init__(self, label_name='labels', logits_name='logits'):
        self.label_name = label_name
        self.logits_name = logits_name
    
    def __call__(self, logits, **kwargs):
        return logits[0]
    
class ZDT3_2():
    
    def __init__(self, label_name='labels', logits_name='logits'):
        self.label_name = label_name
        self.logits_name = logits_name
    
    def __call__(self, logits, **kwargs):
        f1 = logits[0]
        g = 1 + 9 * torch.mean(logits[1:])
        h = 1 - torch.sqrt(f1 / g) - (f1 / g) * torch.sin(10 * math.pi * f1)
        return g * h + 1.