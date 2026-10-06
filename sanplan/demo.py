import argparse, json
from pathlib import Path
from .core import example, greedy, exact, mcts

def main():
    p=argparse.ArgumentParser(); p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--seed',type=int,default=7); p.add_argument('--neural',action='store_true')
    a=p.parse_args(); net=example(); chosen,root=mcts(net,a.iterations,a.seed)
    report={'synthetic':True,'seed':a.seed,'iterations':a.iterations,'budget':net.budget,
            'solutions':{k:{'edges':list(s),'cost':net.cost(s),'served_demand':net.value(s)}
                         for k,s in [('greedy',greedy(net)),('mcts',chosen),('exact',exact(net))]},
            'root_visits':root.visits}
    if a.neural:
        from .neural import train
        model,stats=train(net,a.seed); report['policy_value']=stats
        Path('artifacts').mkdir(exist_ok=True)
        import torch
        torch.save(model.state_dict(),'artifacts/policy-value.pt')
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
