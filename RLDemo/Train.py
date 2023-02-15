import os.path
import os
import torch
import json
from DQNModel import DQN
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DQN_Main:
    def __init__(self):
        self.BUFFER_SIZE = 200000
        self.MIN_REPLAY_SIZE = 50000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GAMMA = 0.99
        self.BATCH_SIZE = 128
        self.EPSILON_START = 0.99
        self.EPSILON_END = 0.1
        self.EPSILON_DECAY = 0.000025
        self.TARGET_UPDATE_FREQ = 10000
        self.LR = 0.00025
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        if os.path.exists('replay_buffer.json'):
            with open("replay_buffer.json", "r") as read_content:
                self.replay_buffer = json.load(read_content)
        else:
                    self.replay_buffer = []

        if os.path.exists('reward_buffer.json'):
            with open("reward_buffer.json", "r") as read_content:
                self.reward_buffer = json.load(read_content)
        else:
            self.reward_buffer = []

        if os.path.exists('step.json'):
            with open("step.json", "r") as read_content:
                self.step = json.load(read_content)
        else:
            self.step = 0

        self.policy_net = DQN(device=self.device).to(self.device)
        self.target_net = DQN(device=self.device).to(self.device)

        if os.path.exists('policy_net.pth'):
            self.policy_net.load_state_dict(torch.load('policy_net.pth'))
            self.target_net.load_state_dict(torch.load('target_net.pth'))
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        if os.path.exists('policy_net_adam.pth'):
            self.optimizer.load_state_dict(torch.load('policy_net_adam.pth'))

        if os.path.exists('loss_hist.json'):
            with open("loss_hist.json", "r") as read_content:
                self.loss_hist = json.load(read_content)
        else:
            self.loss_hist = []

        if os.path.exists('loss_hist_show.json'):
            with open("loss_hist_show.json", "r") as read_content:
                self.loss_hist_show = json.load(read_content)
        else:
            self.loss_hist_show = []

        self.episode_reward = 0
        self.prev_state = None
        self.prev_action = None

    def save_hyperparams(self):
        hyperparams_dict = {
            'BUFFER SIZE': self.BUFFER_SIZE,
            'MIN REPLAY SIZE': self.MIN_REPLAY_SIZE,
            'GAMMA': self.GAMMA,
            'BATCH SIZE': self.BATCH_SIZE,
            'EPSILON START': self.EPSILON_START,
            'EPSILON END': self.EPSILON_END,
            'EPSILON DECAY': self.EPSILON_DECAY,
            'TARGET UPDATE FREQ': self.TARGET_UPDATE_FREQ,
            'LR': self.LR,
        }
        with open("hyperparameters.json", "w") as write:
            json.dump(hyperparams_dict, write)

    def train(self, done):
        # add training step here
        transitions = random.sample(self.replay_buffer, self.BATCH_SIZE)

        states = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        next_states = np.asarray([t[4] for t in transitions])

        states_t = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)

        # compute targets
        _, actions_target = self.policy_net(next_states_t).max(dim=1, keepdim=True)
        target_q_values_1 = self.target_net(next_states_t).gather(dim=1, index=actions_target)
        targets_1 = rewards_t + self.GAMMA * (1 - dones_t) * target_q_values_1

        # compute loss
        q_values = self.policy_net(states_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        # Gradient Descent
        loss = F.mse_loss(action_q_values, targets_1)
        self.loss_hist.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.step % 200 == 0:
            self.loss_hist_show.append(sum(self.loss_hist[-300:])/300)
            self.plot_loss_hist()

        # Update Target Net
        if self.step % self.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # we need to have a done parameter, since we need to save the neural nets if the episode is done
        if done:
            torch.save(self.policy_net.state_dict(), 'policy_net.pth')
            torch.save(self.target_net.state_dict(), 'target_net.pth')
            torch.save(self.optimizer.state_dict(), 'policy_net_adam.pth')
            with open("loss_hist.json", "w") as write:
                json.dump(self.loss_hist, write)
            with open("loss_hist_show.json", "w") as write:
                json.dump(self.loss_hist_show, write)

    def random_action(self):
        return random.choice(self.policy_net.action_space)

    def act(self, state, reward, done, deploy):
        if deploy:
            with torch.no_grad():
                state_t = torch.tensor(state)
                action = self.policy_net.act(state_t)
            return action
        if len(self.replay_buffer) >= self.MIN_REPLAY_SIZE:
            rnd = random.random()
            epsilon = self.EPSILON_START - self.EPSILON_DECAY * self.step
            self.step += 1
            if epsilon < self.EPSILON_END:
                epsilon = self.EPSILON_END
            if rnd <= epsilon:
                action = self.random_action()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state)
                    action = self.policy_net.act(state_t)
        else:
            # fill up the replay buffer
            action = self.random_action()

        if self.prev_state is None:
            # beginning of an episode, we just take the action, nothing to append to the replay buffer
            self.prev_state = state.copy()
            self.prev_action = action

            # we still need to train our neural net here
            if len(self.replay_buffer) >= self.MIN_REPLAY_SIZE:
                # if done neural nets will be saved in the train function
                self.train(done)
            return action
        else:
            # here we add the transitions to replay buffer
            self.episode_reward += reward
            transition = (self.prev_state, self.prev_action, reward, done, state)
            self.replay_buffer.append(transition)
            if len(self.replay_buffer) > self.BUFFER_SIZE:
                self.replay_buffer.pop(0)

            # adjust previous state and action
            self.prev_state = state.copy()
            self.prev_action = action

        if done:
            self.reward_buffer.append(self.episode_reward)
            # since we connect to AnyLogic, we have to save everything every episode
            with open("reward_buffer.json", "w") as write:
                json.dump(self.reward_buffer, write)
            with open("replay_buffer.json", "w") as write:
                json.dump(self.replay_buffer, write)
            with open("step.json", "w") as write:
                json.dump(self.step, write)
            if len(self.reward_buffer)%100 == 0:
                self.plot_reward_buffer()

        if len(self.replay_buffer) >= self.MIN_REPLAY_SIZE:
            # if done neural nets will be saved in the train function
            self.train(done)

        return action

    def plot_reward_buffer(self):
        plt.plot(self.reward_buffer)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig('reward buffer.jpg')
        plt.close()

    def plot_loss_hist(self):
        plt.plot(self.loss_hist_show[10:])
        plt.xlabel('100 Epoch')
        plt.ylabel('Loss')
        plt.savefig('Loss History.jpg')
        plt.close()


if __name__ == '__main__':
    dqn = DQN_Main()
    dqn.save_hyperparams()
