import torch
from datetime import datetime
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from Agent import Agent

TRAIN_MODE = True
EPISODES_NUM = 1750
BANANA_INSTALLATION = "../proj_banan/Banana_Windows_x86_64/Banana.exe"

class ExperienceManager:

    def __init__(self):
        #define the params for later usage
        self.env = None
        self.brain_name = None
        self.agent = None

    #initialize enviroment and set the state space size and  action space size
    def initEnviroment(self):
        print('Initialize env')
        #initalize Unity env
        #update to you
        self.env = UnityEnvironment(file_name=BANANA_INSTALLATION)
        #get the default brain
        self.brain_name = self.env.brain_names[0]
        #reset the environment
        env_info = self.env.reset(train_mode=TRAIN_MODE)[self.brain_name]
        #get size of action and state
        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])
        #initiate Agent
        self.agent = Agent(state_size=self.state_size, action_size=self.action_size, seed=0)
        print('Env init done')

    #run one episode and return total reward
    def runEpisode(self):
       #init score to 0 and reset env
       score = 0
       state = self.env.reset(train_mode=TRAIN_MODE)[self.brain_name].vector_observations[0]
       while True:
            #get greedy action
            action = self.agent.greedy_action(state)
            #perform action
            env_info = self.env.step(action)[self.brain_name]
            #get the step result
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            #store step result and perform learning
            self.agent.step(state, action, reward, next_state, done)
            #update state and score
            state = next_state
            score += reward
            if done:
                #finito
                break
       #return the score of whole episode
       return score

    #run the whole experiments = defined number of episodes
    def runEperiment(self,n_episodes=EPISODES_NUM):
        #init enviroment
        self.initEnviroment()
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        for i_episode in range(1, n_episodes + 1):
            #run one episode
            score =  self.runEpisode()
            #store the score of episode
            scores_window.append(score)
            scores.append(score)
            #print progress
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            #keep progress of last 100 episodes
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        return scores

#store the trained DQN pytorch network
def store_trained_network(exp_manager,filename):
    torch.save(exp_manager.agent.qnetwork_local.state_dict(),filename )

#save the progress of success
def save_plot_scores(scores,filename):
    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #save the plot to file
    plt.savefig(filename + '.png')

#run it all step by step
print('Experiment Starts....')
#Initialize Manager
exp_man = ExperienceManager()
#run experiment
scores = exp_man.runEperiment()
print('Experiment finish')
#get date for storing results
date_sufix = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
#store trained network
store_trained_network(exp_man,'banana_model_'+date_sufix+'.pth')
#store plot of score
save_plot_scores(scores,'score_'+date_sufix+'.png')
print('Store results done')
print('Job done')
