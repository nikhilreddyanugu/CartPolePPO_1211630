import torch.nn as nn 
import torch.nn.functional as F 
class Network(nn.Module): 
   def __init__(self, n_observations=4, n_actions=2): 
      super(Network, self).__init__() 
      self.layer1 = nn.Linear(n_observations, 128) 
      self.layer2 = nn.Linear(128, 128) 
      self.dropout = nn.Dropout()  
      self.layer3 = nn.Linear(128, n_actions) 
   def forward(self, x): 
      x = F.relu(self.layer1(x)) 
      x = self.dropout(x) 
      x = F.relu(self.layer2(x)) 
      return self.layer3(x) 



