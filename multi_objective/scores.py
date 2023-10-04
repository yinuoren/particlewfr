import torch
import numpy as np
import math

from abc import abstractmethod

import objectives as obj
from pytorchltr.evaluation import ndcg

def from_objectives(objectives, train=False, **kwargs):
    if train:
        scores = {
            obj.CrossEntropyLoss: CrossEntropy,
            obj.BinaryCrossEntropyLoss: BinaryCrossEntropy,
            obj.DDPHyperbolicTangentRelaxation: DDP,
            obj.DEOHyperbolicTangentRelaxation: DEO,
            obj.MSELoss: L2Distance,
            obj.InnerProductUtility: InnerProductUtility,
            obj.ListNetLoss: ListNetLoss,
            obj.Fonseca1: Fonseca1,
            obj.Fonseca2: Fonseca2,
            obj.ZDT1_1: ZDT1_1,
            obj.ZDT1_2: ZDT1_2,
            obj.ZDT2_1: ZDT2_1,
            obj.ZDT2_2: ZDT2_2,
            obj.ZDT3_1: ZDT3_1,
            obj.ZDT3_2: ZDT3_2,
            obj.DTLZ7_1: DTLZ7_1,
            obj.DTLZ7_2: DTLZ7_2,
            obj.DTLZ7_3: DTLZ7_3,
        }
    else:
        scores = {
            obj.CrossEntropyLoss: CrossEntropy,
            obj.BinaryCrossEntropyLoss: BinaryCrossEntropy,
            obj.DDPHyperbolicTangentRelaxation: DDP,
            obj.DEOHyperbolicTangentRelaxation: DEO,
            obj.MSELoss: L2Distance,
            obj.InnerProductUtility: NDCG,
            obj.ListNetLoss: NDCG,
        }
    return [scores[o.__class__](label_name = o.label_name, logits_name = o.logits_name, **kwargs) for o in objectives]

class BaseScore():

    def __init__(self, label_name='labels', logits_name='logits', **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class CrossEntropy(BaseScore):

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(logits, labels.long(), reduction='mean').item()


class BinaryCrossEntropy(BaseScore):
    
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]

        if len(logits.shape) > 1 and logits.shape[1] == 1:
            logits = torch.squeeze(logits)

        with torch.no_grad():
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='mean').item()


class NDCG(BaseScore):
    
    def __init__(self, label_name='labels', logits_name='logits', **kwargs):
        super().__init__(label_name=label_name, logits_name=logits_name)
        self.k = kwargs['k_ndcg']
    
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        n = kwargs['n']
        with torch.no_grad():
            return -ndcg(logits, labels, n, k=self.k).mean().item()


class ListNetLoss(BaseScore):
        
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        n = kwargs['n']
        if logits.ndim == 2:
            logits = logits.unsqueeze(-1)
        if labels.ndim == 2:
            labels = labels.unsqueeze(-1)
        
        with torch.no_grad():
            mask = torch.arange(logits.shape[1], device = labels.device)[None, :, None] < n[:, None, None]
        
            masked_logits = torch.where(mask, logits, float('-inf'))
            masked_labels = torch.where(mask, labels, float('-inf'))
            
            logits_smax = torch.nn.functional.softmax(masked_logits, dim=1)
            labels_smax = torch.nn.functional.softmax(masked_labels, dim=1)
            
            return -torch.sum(labels_smax * torch.log(logits_smax + 1e-10), dim=1).mean().item()


class InnerProductUtility():
        
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        n = kwargs['n']
        if logits.ndim == 2:
            logits = logits.unsqueeze(-1)
        if labels.ndim == 2:
            labels = labels.unsqueeze(-1)
            
        with torch.no_grad():
            mask = torch.arange(logits.shape[1], device = labels.device)[None, :, None] < n[:, None, None]
        
            masked_logits = torch.where(mask, logits, float('-inf'))
            
            logits_smax = torch.nn.functional.softmax(masked_logits, dim=1)
            
            return - torch.sum(labels * logits_smax, dim=1).mean().item()


class L2Distance(BaseScore):

    def __call__(self, **kwargs):
        prediction = kwargs['logits']
        labels = kwargs[self.label_name]
        with torch.no_grad():
            return torch.linalg.norm(prediction - labels, ord=2)


class mcr(BaseScore):

    def __call__(self, **kwargs):
        # missclassification rate
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            if len(logits.shape) == 1:
                y_hat = torch.round(torch.sigmoid(logits))
            elif logits.shape[1] == 1:
                # binary case
                logits = torch.squeeze(logits)
                y_hat = torch.round(torch.sigmoid(logits))
            else:
                y_hat = torch.argmax(logits, dim=1)
            accuracy = sum(y_hat == labels) / len(y_hat)
        return 1 - accuracy.item()


class DDP(BaseScore):
    """Difference in Democratic Parity"""

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs['sensible_attribute']
    
        with torch.no_grad():
            n = logits.shape[0]
            logits_s_negative = logits[sensible_attribute.bool()]
            logits_s_positive = logits[~sensible_attribute.bool()]

            return (1/n * torch.abs(torch.sum(logits_s_negative > 0) - torch.sum(logits_s_positive > 0))).cpu().item()


class DEO(BaseScore):
    """Difference in Equality of Opportunity"""

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs['sensible_attribute']

        with torch.no_grad():
            n = logits.shape[0]
            logits_s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
            logits_s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

            return (1/n * torch.abs(torch.sum(logits_s_negative > 0) - torch.sum(logits_s_positive > 0))).cpu().item()        

class Fonseca1(BaseScore):

    @staticmethod
    def f1(theta):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        sum1 = torch.sum((theta - 1.0 / math.sqrt(theta.shape[1])) ** 2, dim=1)
        f1 = 1 - torch.exp(-sum1)
        return f1
    
    def __call__(self, **kwargs):
        return Fonseca1.f1(kwargs[self.logits_name]).cpu().item()
    
    
class Fonseca2(BaseScore):
    
    @staticmethod
    def f2(theta):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        sum2 = torch.sum((theta + 1.0 / math.sqrt(theta.shape[1])) ** 2,dim=1)
        f2 = 1 - torch.exp(-sum2)
        return f2
    
    def __call__(self, **kwargs):
        return Fonseca2.f2(kwargs[self.logits_name]).cpu().item()
    
class ZDT1_1(BaseScore):
    
    def __call__(self, logits, **kwargs):
        return logits[0].item()
    
class ZDT1_2(BaseScore):
    
    def __call__(self, logits, **kwargs):
        g = 1 + 9 * torch.sum(logits[1:]) / (logits.numel() - 1)
        h = 1 - torch.sqrt(logits[0] / g)
        return (g * h).item() 
    
class ZDT2_1(BaseScore):
    
    def __call__(self, logits, **kwargs):
        return logits[0].item()
    
class ZDT2_2(BaseScore):
    
    def __call__(self, logits, **kwargs):
        g = 1 + 9 * torch.sum(logits[1:]) / (logits.numel() - 1)
        h = 1 - (logits[0] / g) ** 2
        return (g * h).item()
    
class ZDT3_1(BaseScore):
    
    def __call__(self, logits, **kwargs):
        return logits[0].item()
    
class ZDT3_2(BaseScore):

    def __call__(self, logits, **kwargs):
        f1 = logits[0]
        g = 1 + 9 * torch.sum(logits[1:]) / (logits.numel() - 1)
        h = 1 - torch.sqrt(f1 / g) - (f1 / g) * torch.sin(10 * math.pi * f1)
        return (g * h).item() + 1.
    
class DTLZ7_1(BaseScore):
    
    def __call__(self, logits, **kwargs):
        return logits[0].item()
    
class DTLZ7_2(BaseScore):
    
    def __call__(self, logits, **kwargs):
        return logits[1].item()
    
class DTLZ7_3(BaseScore):
    
    def __call__(self, logits, **kwargs):
        g = 1 + 9 * torch.mean(logits[2:])
        h = 3 - logits[0] / (1 + g) * (1 + torch.sin(3 * math.pi * logits[0])) - logits[1] / (1 + g) * (1 + torch.sin(3 * math.pi * logits[1]))
        return ((1 + g) * h).item() 