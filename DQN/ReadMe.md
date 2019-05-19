# DQN

## Cart Pole: 

Further details on the CartPole environment: https://gym.openai.com/envs/CartPole-v1/

**Train Command**:

python main.py --train --cuda

The training curve below was smoothened since the original curve is very noisy. You should notice, that in the beginning the agent achieves low rewards, since the actions performed are random but towards the end of the training it should always attain approx. 200 (this solves the environment)

![](images/DQN_CartPole-v0.svg)

**Test Command**:

python main.py --test --cuda


The Episodic rewards attained from this must always be approx. 200
