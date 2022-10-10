
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

SLIDING_WINDOW_FRAME = 20
def plot_for_experiment(scores_q_learn, scores_dqn, scores_dueldqn):
# plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores_q_learn)), scores_q_learn, label="Q Learning", color="b")
    plt.plot(np.arange(len(scores_dqn)), scores_dqn, label="DQN", color="r")
    plt.plot(np.arange(len(scores_dueldqn)), scores_dueldqn, label="Dueling DQN", color="g")
    plt.ylabel('Score')
    plt.xlabel('Episode #') 
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.show()

# plot the average scores with a sliding window
    fig = plt.figure()
    ax = fig.add_subplot(111)

    avgscores_q_learn = sliding_window(scores_q_learn, SLIDING_WINDOW_FRAME)
    plt.plot(np.arange(len(avgscores_q_learn)), avgscores_q_learn, label="Q Learning", color="b")

    
    avgscores_dqn = sliding_window(scores_dqn, SLIDING_WINDOW_FRAME)
    plt.plot(np.arange(len(avgscores_dqn)), avgscores_dqn, label="DQN", color="r")


    avgscores_dueldqn = sliding_window(scores_dueldqn, SLIDING_WINDOW_FRAME)
    plt.plot(np.arange(len(avgscores_dueldqn)), avgscores_dueldqn, label="Dueling DQN", color="g")
    plt.ylabel('Mean Score {}'.format(SLIDING_WINDOW_FRAME))
    plt.xlabel('Episode #') 
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.show()

def plot_training(scores, name):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label=name + "Scores", color="b")
    avg_scores = sliding_window(scores, SLIDING_WINDOW_FRAME)
    plt.plot(np.arange(len(avg_scores)), avg_scores, label=name + "Mean Scores", color="r")
    plt.ylabel('Score')
    plt.xlabel('Episode #') 
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.show()

def sliding_window(scores, window_size):
    
    scores_window = deque(maxlen=window_size)
    avgs_moving_window = []
    for s in scores:
        scores_window.append(s)
        scores_window_avg = np.mean(scores_window)
        avgs_moving_window.append(scores_window_avg)
    return avgs_moving_window