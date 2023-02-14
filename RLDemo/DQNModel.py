from torch import nn
import torch

class DQN(nn.Module):
    def __init__(self, device):
        super(DQN, self).__init__()

        self.action_space = [0, 1, 2, 3, 4, 5]
        self.device = device
        self.num_actions = len(self.action_space)
        self.input_size = 3

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(in_features=32, out_features=self.num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        q_values = self(state_t)
        max_q_index = torch.argmax(q_values).detach().item()
        return max_q_index