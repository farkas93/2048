from game2048.q_learning import Q_agent
from game2048.dqn_learning import DQN_agent
from game2048.dueling_dqn_learning import DuelingDQN_agent
from game2048.show import *
from config import *
from plots import plot_training

def main():

    num_eps = config["EPISODES_TO_TRAIN"] 
    path = config["SAVE_DIR"]
    scores = []
    name = ""

    if config["USE_MODEL"] == Q_LEARN:
      name = "Q Learning"
      # Run the original code of abachurin
      # Run the below line to see the magic. How it starts with random moves and immediately
      # starts climbing the ladder
      agent = Q_agent(n=4, reward=basic_reward, alpha=config["ALPHA"], file=path + "new_agent.npy", savepath=path)

      # Uncomment/comment the above line with the below if you continue training the same agent,
      # update agent.alpha and agent.decay if needed.

      # agent = Q_agent.load_agent(file="best_agent.npy")
      scores = Q_agent.train_run(num_eps, agent=agent, file=path + "new_best_agent.npy", start_ep=0)

    elif config["USE_MODEL"] == DQN:
      name = "DQN"
      agent = DQN_agent(file="dqn_agent.pth", savepath=path)
      scores = DQN_agent.train_run(num_eps, agent=agent, file=path + "dqn_best_agent.pth", start_episode=0)

    else: #config["USE_MODEL"] >= DUELING_DQN
      name = "Dueling DQN"
      agent = DuelingDQN_agent(file="dqn_agent.pth", savepath=path)
      scores =DuelingDQN_agent.train_run(num_eps, agent=agent, file=path + "dqn_best_agent.pth", start_episode=0)

    plot_training(scores, name)

if __name__ == "__main__":
  main()