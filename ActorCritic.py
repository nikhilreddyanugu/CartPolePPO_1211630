import torch.nn as nn 

class ActorCritic(nn.Module): 
    def __init__(self, actor, critic): 
        super(ActorCritic, self).__init__()  # ? Call the parent class constructor first
        self.actor = actor 
        self.critic = critic 

    def forward(self, state): 
        action_pred = self.actor(state) 
        value_pred = self.critic(state) 
        return action_pred, value_pred





