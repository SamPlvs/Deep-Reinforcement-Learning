# DQN

## Cart Pole: 

Further details on the CartPole environment: https://gym.openai.com/envs/CartPole-v1/

### Non-GPU Holders:
if you do not have a GPU please run the simple DQN file (also tested on CartPole), filename: DQN_simple_all.py
Special thanks to TianhongDai for creating this, (https://github.com/TianhongDai)

### GPU Holders:
**Train Command**:

python main.py --train --cuda

The training curve below was smoothened since the original curve is very noisy. You should notice, that in the beginning the agent achieves low rewards, since the actions performed are random but towards the end of the training it should always attain approx. 200 (this solves the environment)

![](images/DQN_CartPole-v0.svg)

**Test Command**:

python main.py --test --cuda

Though do keep in mind, you'll have to change the default input for the argument: '--load_model' to the path where you've stored the model. The Episodic rewards attained from this must always be approx. 200
