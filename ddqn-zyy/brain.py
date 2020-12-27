import random
import math
import numpy as np

from memory import ReplayMemory
from memory import Transition
from models import DQN
from params import *

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T


class Brain(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.memory = ReplayMemory(CAPACITY)
        self.steps_done = 0


        self.main_net = DQN(n_actions=4).to(device)
        self.target_net = DQN(n_actions=4).to(device)
        
        
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        print(self.main_net)

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)


    def decide_action(self, state, n_steps, mode='train'):
        epsilon = 0.5 * (1 / (n_steps + 1))

        if epsilon <= np.random.uniform(0, 1) or mode == 'test':
            self.main_net.eval()
            with torch.no_grad():
                action = self.main_net(state.to('cuda')).max(1)[1].view(1,1)

        else:
            action = torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

        return action
        

    def replay(self):
        """
        经验回放更新网络参数
        """
        # 检查经验池数据量是否够一个批次
        if len(self.memory) < BATCH_SIZE:
            return
        
        # 创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 找到
        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_network()

    def make_minibatch(self):
        """创建小批量数据"""

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None]).to('cuda')
        
        state_batch = torch.cat(batch.state).to('cuda')
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        """获取期望的Q值"""

        self.main_net.eval()
        self.target_net.eval()
        
        self.state_action_values = self.main_net(self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, self.batch.next_state)),
            device=device, dtype=torch.bool)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        next_state_values[non_final_mask] = self.target_net(self.non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + self.reward_batch
        return expected_state_action_values

    def update_main_network(self):
        """更新网络参数"""

        # 将网络切换训练模式
        self.main_net.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()

        loss.backward()
        
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
    
    def save_net_state(self, filename):
        torch.save(self.main_net, filename)
    
    def load_net_state(self, filename):
        self.main_net = torch.load(filename)
        self.target_net.load_state_dict(self.main_net.state_dict())