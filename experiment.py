from cProfile import label
from turtle import color
from collections import deque

from matplotlib.pyplot import plot
from game2048.q_learning import Q_agent
from game2048.dqn_learning import DQN_agent
from game2048.dueling_dqn_learning import DuelingDQN_agent
from game2048.show import *
from config import *
from plots import plot_for_experiment

def main():

    num_eps = config["EPISODES_TO_TRAIN"] 

    # Train a Dueling DQN
    path = "./saved_data/duel_dqn/"
    agent = DuelingDQN_agent(file="dqn_agent.pth", savepath=path)
    scores_dueldqn= DuelingDQN_agent.train_run(num_eps, agent=agent, file=path + "dqn_best_agent.pth", start_episode=0)

    # Train a DQN
    path = "./saved_data/dqn/" 
    agent = DQN_agent(file="dqn_agent.pth", savepath=path)
    scores_dqn = DQN_agent.train_run(num_eps, agent=agent, file=path + "dqn_best_agent.pth", start_episode=0)


    #Train the Q Learning algo of abachurin
    path = "./saved_data/q_learning/"
    agent = Q_agent(n=4, reward=basic_reward, alpha=config["ALPHA"], file=path + "new_agent.npy", savepath=path)
    scores_q_learn = Q_agent.train_run(num_eps, agent=agent, file=path + "new_best_agent.npy", start_ep=0)

    plot_for_experiment(scores_q_learn, scores_dqn, scores_dueldqn)


if __name__ == "__main__":
  main()