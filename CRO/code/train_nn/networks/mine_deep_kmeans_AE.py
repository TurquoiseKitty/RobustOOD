import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import logging
import ipdb

class Encoder(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()

        self.rep_dim = output_size

        self.fc2 = nn.Linear(input_size, 10)

        self.fc4 = nn.Linear(10, output_size)

        nn.init.uniform_(self.fc2.weight, -1.,1.)
        # nn.init.uniform_(self.fc3.weight, 0.,0.5)
        nn.init.uniform_(self.fc4.weight, -1.,1.)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class Decoder(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(output_size, 10)
        # self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.fc3 = nn.Linear(10, input_size)

        nn.init.uniform_(self.fc1.weight, -1.,1.)
        # nn.init.uniform_(self.fc2.weight, 0.,0.5)
        nn.init.uniform_(self.fc3.weight, -1.,1.)

    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
#         out = F.relu(self.fc4(out))
        out = out.view(out.size(0), -1)
        return out

class KMeansCriterion(nn.Module):
    
    def __init__(self, lmbda):
        super().__init__()

        self.lmbda = lmbda
    
    def forward(self, embeddings, centroids):
        distances = self.lmbda*torch.sum(torch.abs(embeddings[:, None, :] - centroids)**2, 2)

        cluster_distances, cluster_assignments = distances.min(1)
        loss = self.lmbda * cluster_distances.sum()
        return loss, cluster_assignments.detach()
    
    

def block_net(in_f, out_f,h1,h2):
    return nn.Sequential(
        nn.Linear(in_f, h1, bias=False) ,
        nn.ReLU(),
        nn.Linear(h1, h2, bias=False) ,
        nn.ReLU(),
        nn.Linear(h2, out_f, bias=False) 
    )
class main_net_AE(BaseNet):

    def __init__(self,n_class,main_size,out_size):
        super().__init__()
        logger = logging.getLogger()
        rep = out_size
        self.rep_dim = rep
        in_f=main_size
        out_f=rep
        # self.trace = []
        self.net = block_net(in_f, out_f,10,8)
        self._init_weights(self.net)
        # logger.info(self.net_blocks)

        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        return self.net(x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight,0.,1.)

        



