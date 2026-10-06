import unittest
from sanplan.core import example, mcts, exact
class Tests(unittest.TestCase):
    def test_search_is_feasible_and_bounded(self):
        net=example(); s,root=mcts(net,200,7)
        self.assertLessEqual(net.cost(s),net.budget)
        self.assertLessEqual(net.value(s),net.value(exact(net)))
        self.assertEqual(root.visits,200)
        reached={0}
        for _ in s:
            for i in s:
                u,v,_=net.edges[i]
                if u in reached or v in reached: reached.update((u,v))
        self.assertEqual(reached,net.nodes(s))
    def test_invalid_action(self):
        with self.assertRaises(ValueError): example().step((),4)
