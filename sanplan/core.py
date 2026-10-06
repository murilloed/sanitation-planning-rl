"""Connected expansion, four-phase MCTS, and exact small-instance oracle."""
from dataclasses import dataclass, field
import math
import random

@dataclass(frozen=True)
class Network:
    edges: tuple  # (u, v, cost)
    demand: tuple
    budget: int

    def nodes(self, chosen):
        nodes = {0}
        for i in chosen:
            nodes.update(self.edges[i][:2])
        return nodes

    def cost(self, chosen):
        return sum(self.edges[i][2] for i in chosen)

    def value(self, chosen):
        return sum(self.demand[i] for i in self.nodes(chosen))

    def actions(self, chosen):
        nodes = self.nodes(chosen)
        return [i for i, (u, v, c) in enumerate(self.edges)
                if i not in chosen and c + self.cost(chosen) <= self.budget
                and ((u in nodes) != (v in nodes))]

    def step(self, chosen, action):
        if action not in self.actions(chosen):
            raise ValueError('Action disconnected, repeated or over budget')
        return tuple(sorted((*chosen, action)))

def example():
    return Network(((0,1,3),(0,2,2),(1,3,3),(2,4,2),(4,5,2),(3,5,4)),
                   (0,30,10,70,20,80), 8)

def greedy(net):
    state = ()
    while net.actions(state):
        a = max(net.actions(state), key=lambda i:
                (net.value(net.step(state,i))-net.value(state))/net.edges[i][2])
        state = net.step(state, a)
    return state

def exact(net, start=()):
    cache = {}
    def solve(state):
        if state not in cache:
            candidates = [state] + [solve(net.step(state,a)) for a in net.actions(state)]
            cache[state] = max(candidates, key=lambda s:(net.value(s), -net.cost(s)))
        return cache[state]
    return solve(start)

@dataclass
class Node:
    state: tuple
    parent: object = None
    children: list = field(default_factory=list)
    visits: int = 0
    total: float = 0.0

def mcts(net, iterations=200, seed=7):
    if iterations < 1:
        raise ValueError('iterations must be positive')
    rng, root = random.Random(seed), Node(())
    best = ()
    for _ in range(iterations):
        node = root
        # Selection: descend only when all legal actions have been expanded.
        while node.children and len(node.children) == len(net.actions(node.state)):
            node = max(node.children, key=lambda n:n.total/n.visits +
                       math.sqrt(2*math.log(max(1,node.visits))/n.visits))
        # Expansion: one previously unseen legal successor.
        states = {c.state for c in node.children}
        unseen = [net.step(node.state,a) for a in net.actions(node.state)
                  if net.step(node.state,a) not in states]
        if unseen:
            child = Node(rng.choice(unseen), node)
            node.children.append(child)
            node = child
        # Simulation: stochastic rollout until no connected action fits.
        state = node.state
        while net.actions(state):
            state = net.step(state, rng.choice(net.actions(state)))
        if (net.value(state), -net.cost(state)) > (net.value(best), -net.cost(best)):
            best = state
        reward = net.value(state) / max(1,sum(net.demand))
        # Backpropagation of the same single-agent return.
        while node is not None:
            node.visits += 1
            node.total += reward
            node = node.parent
    return best, root
