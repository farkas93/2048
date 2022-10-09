

config = {}
config["EPISODES_TO_TRAIN"] = 200
config["ALPHA"] = 0.1
config["USE_DQN"] = True
config["USE_ABACHRIN"] = False
config["SAVE_DIR"] = "./saved_data/q_learning/"
if config["USE_DQN"] :
  config["SAVE_DIR"] = "./saved_data/dqn/"
if config["USE_ABACHRIN"]:
  config["SAVE_DIR"] = "./saved_data/abachurin/"
