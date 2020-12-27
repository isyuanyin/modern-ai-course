import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import torch

import gym

from wrappers import *
from utils import *
from agent import *
from params import *

class Environment:
    """Pong 游戏执行环境"""

    def __init__(self, save_path='ddqn_pong_model'):
        self.env = gym.make("PongNoFrameskip-v4")
        self.env = make_env(self.env)

        n_actions = self.env.action_space.n

        self.agent = Agent(n_actions)

        self.save_path = save_path

        self.n_steps = 0

    def obs2state(self, obs):
        """ 
        观察值转换成状态
        """

        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self, n_episodes, render=False):
        """
        训练Agent
        """
        mean_rewards = []
        episode_reward = []

        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.obs2state(obs)

            # 记录 reward
            total_reward = 0.0
            for t in count():
                self.n_steps += 1
                action = self.agent.get_action(state, self.n_steps)

                if render:
                    self.env.render()

                obs, reward, done, info = self.env.step(action)

                total_reward += reward
                
                if not done:
                    next_state = self.obs2state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=device)

                self.agent.memorize(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state

                
                if self.n_steps > INITIAL_SIZE:
                    self.agent.update()

                    if self.n_steps % TARGET_UPDATE == 0:
                        self.agent.update_target()

                if done:
                    break

            if (episode + 1 ) % 20 == 0:
                    print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.n_steps, episode+1, n_episodes, total_reward))
            

            # 计算平均 reward
            episode_reward.append(total_reward)
            mean_100ep_reward = round(np.mean(episode_reward[-100:]), 1)
            mean_rewards.append(mean_100ep_reward)
            
        self.env.close()
        plot_learning_curve(mean_rewards, filename=self.save_path + '.png' )
        
        # 保存模型
        self.agent.save_model(self.save_path)
        return


    def test(self, n_episodes=1, render=True):


        # 加载模型
        self.agent.load_model(self.save_path)

        dir_suffix = str(time.monotonic())
        env = gym.wrappers.Monitor(self.env, './videos/' + dir_suffix)

        for episode in range(n_episodes):
            obs = env.reset()
            state = self.obs2state(obs)
            total_reward = 0.0

            for t in count():
                action = self.agent.get_action(state, episode, mode='test')
                if render:
                    env.render()
                    time.sleep(0.02)

                obs, reward, done, info = env.step(action)

                total_reward += reward

                if not done:
                    next_state = self.obs2state(obs)
                else:
                    next_state = None

                state = next_state

                if done:
                    print("Finished Episode {} with reward {}".format(episode, total_reward))
                    break

        env.close()
        return

    def run(self, train_episodes, test_episodes, render=True):
        self.train(train_episodes)
        self.test(test_episodes, render)
############################################
################## Main ####################
############################################
import sys
if __name__ == '__main__':
    # 超参数

    model_save_path = 'ddqn' + '_episode' +str(N_EPISODES) + '_lr' + str(lr) \
                        + '_batch' + str(BATCH_SIZE)
    pong_env = Environment(model_save_path)

    if len(sys.argv) <= 1 or sys.argv[1] == 'run':
        pong_env.run(train_episodes=N_EPISODES, test_episodes=1, render=True)

    elif sys.argv[1] == 'train':
        pong_env.train(N_EPISODES, render=False)

    elif sys.argv[1] == 'test':
        pong_env.test(n_episodes=1, render=True)

    else:
        print("Enter correct mode name please.")