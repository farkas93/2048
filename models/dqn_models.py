import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class ConvDuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ConvDuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.fc1 = nn.Linear(256, fc1_units)
        self.fc2 = nn.Linear(256, fc1_units)
        self.action_advantages = nn.Linear(fc1_units, action_size)
        self.state_value = nn.Linear(fc1_units, 1)

        self.out_layer = nn.Linear(action_size+1, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv1(state)
        x = self.conv2(x)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x))
        x_1 = F.relu(self.action_advantages(x_1))
        x_2 = F.relu(self.state_value(x_2))
        x = torch.cat([x_1, x_2], dim=1)
        return self.out_layer(x)   

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(state_size, fc1_units)
        self.action_advantages = nn.Linear(fc1_units, action_size)
        self.state_value = nn.Linear(fc1_units, 1)

        self.out_layer = nn.Linear(action_size+1, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x_1 = F.relu(self.fc1(state))
        x_2 = F.relu(self.fc2(state))
        x_1 = F.relu(self.action_advantages(x_1))
        x_2 = F.relu(self.state_value(x_2))
        x = torch.cat([x_1, x_2], dim=1)
        return self.out_layer(x)        