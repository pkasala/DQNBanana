[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[myagent]: http://www.youtube.com/watch?v=liwftuF-CC4
# Navigation Project - Eat Bananas 
### Introduction
This project, describes the reinforcement learning technic for resolve the navigation problem. 
The problem is describe with continues state space and descret action space.
The goal is simple, the monkey should eat only yellow bananas :) 
I used Double DQN with Priority Experience replay alogorithm.
The enviroment comes from Unity, please read the Unity Environment, before making a copy and trying yourself!

### Get Started 
Dowload the zip file navigation.zip and start with ExperienceManager.py

### Enviroment description
**See trained agent in action [here](http://www.youtube.com/watch?v=liwftuF-CC4)**

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average *score of +13 over 100* consecutive episodes.

### Project structure
The project in writen in python and using the pytorch framework for Deep Neural network. More requirements read bellow.
The project files are following:
- ExperienceManager.py - the main file.  Responsible for run the experience, run episode, interact with agent and store statistic of episode reward 
- Navigation.ipynb - copy of ExperienceManager.py but in Jupyter Notebook
- Agent.py - responsible for choosing action in particular state, interact with memory, and learning process             
- PriorityMemory.py - class is reponsible for storing data in SumTree data structure, and sampling  data from it
- SumTree.py - data structure defined for storing experiences in ordered tree. The branch is the value of sum of priorities under this branch. And the leaf is experience. Search over tree is than based on priority value.
- Model.py - define the Neural network model in Pytorch


- model.pth - the learned neural network model
- Report.pdf - describe the work process and some hyper parameter testing

### Installation requirement
The project was tested on 3.6 python and requires the following packages to be installed:
- numpy 1.16.4
- torch 1.1.0
- matplotlib 3.1.0
- unityagent 0.4.0

### Unity Environment
After success instalation please navigate to line num 25 in ExperienceManager.py and update the path to your installation directory
To try it yourself and see how wise you agent can be :), you'll need to download a new Unity environment.
You need only select the environment that matches your operating system:

* Linux: [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
* Mac OSX: [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
* Windows (32-bit): [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
* Windows (64-bit): [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

