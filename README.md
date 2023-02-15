# RL in Anylogic Demo
This is a simple demo of implementation of RL in AnyLogic based on a simplified version of OpenAI Gym Taxi-v3.

### Motivation
As there is currently no stable AnyLogic library for stable RL training with Python, this AnyLogic model demonstrates a stable way to train RL agent with the use of Pypeline library.

### Content
The RLDemo.alp file is the AnyLogic model for the simplified Taxi-v3 environment, and it is made to support communication with Python for RL training in the simulated environment. The two Python files, Train.py and DQNModel.py, are python code for RL training. The JSON and PTH files are locally saved information for RL training. 

### Description of Environment

This environment is in a 4*4 grid world, where there is an RL controlled taxi and a passenger. A visualization of the grid world is shown in the figure below, where the green lines represent walls that the taxi cannot go across. The initial location of the passenger is G, and the destination of the passenger is Y. The taxi will be initialized anywhere randomly other than the passenger location. The goal of the taxi is to first pick up the passenger and then drop the passenger off at the destination. Once the passenger is dropped off or more than 200 action steps are taken, the episode ends. The action space in this environment is 0: move up, 1: move down, 2: move left, 3: move right, 4: pick up, and 5: drop off. The state space is the position of taxi on x-axis, the position of taxi on y-axis, and whether the passenger has been picked up (0 or 1). When the taxi makes a failed pick up or drop off, it receives a reward of -10. When the taxi successfully drops off the passenger, it receives a reward of +20. The taxi receives a -1 reward, unless one of the above-mentioned rewards is triggered.

![Taxi](https://user-images.githubusercontent.com/62451645/218914898-4b4a75b4-e41e-4202-a6e0-d3a2511e8939.png)

### Video Demonstration
Below is a short video to demonstrate the performance of the trained RL agent, where yellow is the taxi, red is the passenger, and green is the destination. The taxi controlled by the trained RL agent picked up the passenger and dropped off the passenger at destination without any extra steps. It received the maximum reward possible.

https://user-images.githubusercontent.com/62451645/218914922-a6ad3d8e-d2a7-4e7d-bb34-751c7b010ff1.mp4


