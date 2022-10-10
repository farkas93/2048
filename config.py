"""
MODEL ENUMS
Options for the USE_MODEL Parameter
"""
ABACHURIN = 1
Q_LEARN = 2
DQN = 3
DUELING_DQN = 4

"""
App Config
"""
config = {}
config["EPISODES_TO_TRAIN"] = 200
config["ALPHA"] = 0.1
config["USE_MODEL"] = DUELING_DQN  

if config["USE_MODEL"] == ABACHURIN:
  config["SAVE_DIR"] = "./saved_data/abachurin/"
elif config["USE_MODEL"] == Q_LEARN:
  config["SAVE_DIR"] = "./saved_data/q_learning/"
elif config["USE_MODEL"] == DQN:
  config["SAVE_DIR"] = "./saved_data/dqn/"
else: #Assume to be DUELING_DQN
  config["SAVE_DIR"] = "./saved_data/duel_dqn/"