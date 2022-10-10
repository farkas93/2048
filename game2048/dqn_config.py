
from game2048.game_logic import *

def basic_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return next_game.score - game.score

def odometer_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return (next_game.score - game.score) * game.odometer


def log_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return np.log(next_game.score + 1) - np.log(game.score + 1)
    

config = {}
config["REWARD_FUNCTION"] = log_reward
config["BUFFER_SIZE"] = int(1e6)  # replay buffer size
config["BATCH_SIZE"] = 32         # minibatch size
config["GAMMA"] = 0.99            # discount factor
config["TAU"]= 1e-3              # for soft update of target parameters
config["LR"] = 0.00025             # learning rate 
config["UPDATE_EVERY"] = 4        # how often to update the network
config["EPOCH_SIZE"] = 50        # how many episodes are an epoch
config["EARLY_OUT"] = 3           # Number of epochs we allow to have stagnation in learning before early out.

config["EPSILON"] = 1.0                # Epsilon value for the epsilon greedy policy
config["EPS_DECAY"] = 0.9
config["EPS_MIN"] = 0.0001