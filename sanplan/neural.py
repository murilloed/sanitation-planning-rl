"""Supervised policy/value model. This version does not guide MCTS yet."""
import torch
from torch import nn
from .core import exact

class PolicyValue(nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.input = nn.Linear(edges + 1, 32)
        self.residual = nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,32))
        self.policy = nn.Linear(32, edges)
        self.value = nn.Linear(32,1)

    def forward(self, x):
        h = torch.relu(self.input(x))
        h = torch.relu(h + self.residual(h))
        return self.policy(h), torch.sigmoid(self.value(h)).squeeze(-1)

def encode(net, state):
    return [float(i in state) for i in range(len(net.edges))] + [
        (net.budget-net.cost(state))/net.budget]

def train(net, seed=7, epochs=150):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    states, seen = [()], {()}
    for s in states:
        for a in net.actions(s):
            n = net.step(s,a)
            if n not in seen:
                states.append(n); seen.add(n)
    active = [s for s in states if net.actions(s)]
    labels = [max(net.actions(s),key=lambda a:net.value(exact(net,net.step(s,a)))) for s in active]
    x = torch.tensor([encode(net,s) for s in active])
    y = torch.tensor(labels)
    values = torch.tensor([net.value(exact(net,s))/sum(net.demand) for s in active])
    mask = torch.tensor([[i in net.actions(s) for i in range(len(net.edges))] for s in active])
    model = PolicyValue(len(net.edges))
    opt = torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(epochs):
        logits, v = model(x)
        loss = nn.functional.cross_entropy(logits.masked_fill(~mask,-1e9),y) + nn.functional.mse_loss(v,values)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, {'training_states':len(active),'training_loss':float(loss.detach()),
                   'evaluation':'training fit only; no out-of-instance generalization claim'}
