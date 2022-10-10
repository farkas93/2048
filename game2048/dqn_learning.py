from matplotlib.cbook import flatten
from game2048.game_logic import *
import random
from collections import deque, namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from models.dqn_models import *


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 16         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
EPOCH_SIZE = 100        # how many episodes are an epoch
EARLY_OUT = 3           # Number of epochs we allow to have stagnation in learning before early out.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def basic_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return next_game.score - game.score

# The RL agent. It is not actually Q, as it tries to learn values of the states (V), rather than actions (Q).
# Not sure what is the correct terminology here, this is definitely a TD(0), basically a modified Q-learning.
# The important details:
# 1) The discount parameter gamma = 1. Don't see why discount rewards in this episodic task.
# 2) Greedy policy, epsilon = 0, no exploration. The game is pretty stochastic as it is, no need.
# 3) The valuation function is basically just a linear operator. I takes a vector of the values of
#    1114112 (=65536 * 17) features and dot-product it by the vector of 1114122 weights.
#    Sounds like a lot of computation but! and this is the beauty - all except 17 of the features
#    are exactly zero, and those 17 are exactly 1. So the whole dot product is just a sum of 17 weights,
#    corresponding to the 1-features.
# 4) The same goes for back-propagation. We only need to update 17 numbers of 1m+ on every step.
# 5) But in fact we update 17 * 8 weights using an obvious D4 symmetry group acting on the board

class DQN_agent:
    #TODO: REWRITE THIS PART TO WORK WITH A DQN approach
    save_file = "agent.pth"     # saves the weights, training step, current alpha and type of features

    def __init__(self, reward=basic_reward,
                 file=None, seed=4, savepath=""):

        """Initialize an Agent object.       
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.R = reward
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.file = savepath + file or DQN_agent.save_file
        self.savepath = savepath

        #DQN code

        self.state_size = 16
        self.action_size = 4
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        summary(self.qnetwork_local, (self.state_size,))

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # a numpy.save method works fine not only for numpy arrays but also for ordinary lists
    def save_agent(self, file=None):
        file = file or self.file
        torch.save(self.qnetwork_local.state_dict(), file)
        pass

    @staticmethod
    def load_agent(file=save_file):
        agent = DQN_agent()
        self.qnetwork_local.load_state_dict(torch.load(save_file))
        return agent

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # The game 2048 has two kinds of states. After we make a move - this is the one we try to evaluate,
    # and after the random 2-4 tile is placed afterwards.
    # On each step we check which of the available moves leads to a state, which has the highest value
    # according to the current weights of our evaluator. Now we use that best value, our learning rate
    # and the usual Bellman Equation to make a back-propagation update for the previous such state.
    # In this case - we adjust several weights by the same small delta.
    # A very fast and efficient procedure.
    # Then we move in that best direction, add random tile and proceed to the next cycle.

    def train_episode(self, eps):
        game = Game()
        state, action = None, 0
        nr_moves = 0 
        while not game.game_over():
            nr_moves += 1
            state = game.row.flatten()
            action = int(self.act(state, eps))
            
            game.move(action)
            game.new_tile()
            next_state = game.row.flatten()

            reward = self.R(game, action)
            self.step(state, action, reward, next_state, game.game_over())
        game.history.append(game)
        return game

    # We save the agent every 100 steps, and best game so far - when we beat the previous record.
    # So if you train it and have to make a break at some point - no problem, by loading the agent back
    # you only lose last <100 episodes. Also, after reloading the agent one can adjust the learning rate,
    # decay of this rate etc. Helps with the experimentation.
    @staticmethod
    def train_run(num_eps, agent=None, file=None, start_episode=0, saving=True, 
                eps_start=0.0001, eps_end=0.0001, eps_decay=0.5):
        if agent is None:
            agent = DQN_agent()
        if file:
            agent.file = file
        av1000 = []
        ma100 = []
        reached = [0] * 7
        best_game, best_score = None, 0
        start = time.time()

        eps = eps_start  
        for i in range(start_episode + 1, num_eps + 1):
            game = agent.train_episode(eps)
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            ma100.append(game.score)
            av1000.append(game.score)
            if game.score > best_score:
                best_game, best_score = game, game.score
                print('new best game!')
                print(game)
                if saving:
                    game.save_game(file=agent.savepath + 'best_game.npy')
                    print('game saved at best_game.npy')
            max_tile = np.max(game.row)
            if max_tile >= 10:
                reached[max_tile - 10] += 1
            if i - start_episode > EPOCH_SIZE:
                ma100 = ma100[1:]
            print(i, game.odometer, game.score, 'reached', 1 << np.max(game.row), '100-ma=', int(np.mean(ma100)))
            if saving and i % EPOCH_SIZE == 0:
                agent.save_agent()
                print(f'agent saved in {agent.file}')
            if i % 1000 == 0:
                print('------')
                print((time.time() - start) / 60, "min")
                start = time.time()
                print(f'episode = {i}')
                print(f'average over last 1000 episodes = {np.mean(av1000)}')
                av1000 = []
                for j in range(7):
                    r = sum(reached[j:]) / 10
                    print(f'{1 << (j + 10)} reached in {r} %')
                reached = [0] * 7
                print(f'best score so far = {best_score}')
                print(best_game)
                print(f'current learning rate = {LR}')
                print('------')

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)