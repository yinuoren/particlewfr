import torch.nn as nn
import torch

class MultiLeNet(nn.Module):


    def __init__(self, dim, **kwargs):
        super().__init__()
        self.shared =  nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(720 , 50),
            nn.ReLU(),
        )
        self.private_left = nn.Linear(50, 10)
        self.private_right = nn.Linear(50, 10)
    

    def forward(self, batch):
        x = batch['data']
        x = self.shared(x)
        return dict(logits_l=self.private_left(x), logits_r=self.private_right(x))


    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']


class MultiLeNetCondition(nn.Module):
    def __init__(self, dim, n_objectives, **kwargs):
        super().__init__()
        self.film_generator = nn.Sequential(
            nn.Linear(n_objectives, 20),
            nn.ReLU(),
            nn.Linear(20, 60)
        )

        self.shared1 = nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.shared2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.shared3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(720, 50),
            nn.ReLU(),
        )
        self.private_left = nn.Linear(50, 10)
        self.private_right = nn.Linear(50, 10)
    

    def forward(self, batch):
        x = batch['data']
        r = batch['alpha']

        film_vector = self.film_generator(r)

        x = self.shared1(x)
        beta1 = film_vector[:10].view(1, 10, 1, 1)
        gamma1 = film_vector[10:20].view(1, 10, 1, 1)
        x = x * beta1 + gamma1


        x = self.shared2(x)
        beta2= film_vector[20:40].view(1, 20, 1, 1)
        gamma2 = film_vector[40:60].view(1, 20, 1, 1)
        x = x * beta2 + gamma2

        x = self.shared3(x)

        return dict(logits_l=self.private_left(x), logits_r=self.private_right(x))


    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']


class FullyConnected(nn.Module):
    
    @staticmethod
    def _create_layers(arch):
        layers = []
        for i in range(len(arch)-2):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(arch[-2], arch[-1]))
        return layers

    def __init__(self, dim, **kwargs):
        super().__init__()
        arch = kwargs['arch']
        # self.f = nn.Sequential(
        #     nn.Linear(dim[0], 60),
        #     nn.ReLU(),
        #     nn.Linear(60, 25),
        #     nn.ReLU(),
        #     nn.Linear(25, 1),
        # )
        self.f = nn.Sequential(*self._create_layers([dim[0]] + arch + [1]))

    def forward(self, batch):
        x = batch['data']
        return dict(logits=self.f(x))


class FullyConnectedExplicit(nn.Module):
    
    @staticmethod
    def _create_layers(arch):
        layers = []
        for i in range(len(arch)-2):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(arch[-2], arch[-1]))
        return layers

    def __init__(self, dim, param_dim, **kwargs):
        super().__init__()
        arch = kwargs['arch']
        self.f = nn.Sequential(*self._create_layers([dim[0]] + arch + [param_dim]))

    def forward(self, batch):
        x = batch['alpha']
        logits = self.f(x)
        if 'clip' in batch:
            a, b = batch['clip'][0], batch['clip'][1]
            logits = a + (b - a) * torch.sigmoid(10 * logits)
        return dict(logits=logits)


class FullyConnectedCondition(nn.Module):

    @staticmethod
    def _create_layers(arch):
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            layers.append(nn.ReLU())
        return layers

    def __init__(self, dim, n_objectives, **kwargs):
        super().__init__()
        arch = kwargs['arch']
        
        self.fc_layers = nn.ModuleList(self._create_layers([dim[0]] + arch))
        self.last_layer = nn.Linear(arch[-1], 1)
        
        film_length = sum(arch) * 2
        
        self.film_generator = nn.Sequential(
            nn.Linear(n_objectives, 20),
            nn.ReLU(),
            nn.Linear(20, film_length)
        )

    def forward(self, batch):
        x = batch['data']
        r = batch['alpha']

        film_vector = self.film_generator(r)
        
        start_idx = 0
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                
                beta = film_vector[start_idx:start_idx+out_features]
                gamma = film_vector[start_idx+out_features:start_idx+2*out_features]
                
                start_idx += 2 * out_features  # Increment starting index for the next set of beta, gamma
                
                x = layer(x)
                x = x * beta + gamma  # Apply FiLM
                
            else:
                x = layer(x)
                
        x = self.last_layer(x)

        return dict(logits=x)
    
    
class Dummy(nn.Module):
    
    def __init__(self, param_dim, initialization, **kwargs):
        super().__init__()
        if initialization == 'zero':
            self.param = nn.Parameter(torch.zeros(param_dim))
        else:
            self.param = nn.Parameter(torch.rand(param_dim))
        
    def forward(self, batch):
        logits = self.param
        if 'clip' in batch:
            a, b = batch['clip'][0], batch['clip'][1]
            logits = a + (b - a) * torch.sigmoid(10 * logits)
        return dict(logits=logits)