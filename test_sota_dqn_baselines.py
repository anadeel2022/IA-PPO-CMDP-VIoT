import numpy as np
import torch
from pathlib import Path

from hiot_sota_dqn import (
    GreedyQPolicy, StandardQNet, ValueDQNConfig, train_value_dqn,
    save_value_policy, load_value_policy,
)
from viot_agentic_runner import UNIVERSAL_VALUE_SCHEMES, UNIVERSAL_LEARNED_SCHEMES


class TinyEnv:
    n_actions=4; n_classes=3; n_channels=1; slot_s=0.001
    def __init__(self): self.t=0; self.agentic_episode_summaries=[]
    def _make_obs(self): return np.array([self.t/10.0, 1.0, 0.0],dtype=np.float32)
    def reset(self): self.t=0; return self._make_obs()
    def get_feasible_action_mask(self): return np.array([True,False,True,False])
    def get_action_mask(self): return self.get_feasible_action_mask()
    def step(self,a):
        assert self.get_feasible_action_mask()[int(a)]
        self.t += 1
        r=1.0 if int(a)==2 else 0.0
        info={
            'delivered_bits':1000.0 if a==2 else 0.0,'blocked':0,'preempt':0,'arrivals':1,
            'completed_requests':1 if a==2 else 0,'request_conservation_error':0,
        }
        return self._make_obs(),r,self.t>=5,info


def test_value_schemes_are_universal():
    assert UNIVERSAL_VALUE_SCHEMES == {'SOTA-DQN','SOTA-DuelingDDQN'}
    assert UNIVERSAL_VALUE_SCHEMES.issubset(UNIVERSAL_LEARNED_SCHEMES)


def test_greedy_policy_respects_mask():
    net=StandardQNet(3,4,8)
    with torch.no_grad():
        for p in net.parameters(): p.zero_()
        net.net[-1].bias.copy_(torch.tensor([0.0,10.0,2.0,9.0]))
    pol=GreedyQPolicy(net,torch.device('cpu'),'standard',3,4,8)
    assert pol.act(np.array([0,0,0],dtype=np.float32),action_mask=[1,0,1,0])==2


def test_train_both_value_baselines_and_checkpoint(tmp_path: Path):
    cfg=ValueDQNConfig(batch_size=4,replay_capacity=100,warmup_steps=4,train_every=1,target_sync_steps=3,eps_fraction=.5,hidden_dim=16)
    for arch,double in [('standard',False),('dueling',True)]:
        out=train_value_dqn(TinyEnv(),episodes=4,steps_per_ep=5,seed=7,device='cpu',architecture=arch,double_dqn=double,cfg=cfg)
        pol=out['policy']
        assert pol.act(np.array([0,1,0],dtype=np.float32),action_mask=[1,0,1,0]) in (0,2)
        p=tmp_path/f'{arch}.pt'; save_value_policy(pol,p,{'x':1})
        loaded,meta=load_value_policy(p,'cpu')
        assert meta['x']==1
        assert loaded.act(np.array([0,1,0],dtype=np.float32),action_mask=[1,0,1,0]) in (0,2)
