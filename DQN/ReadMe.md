# DQN
---
The algorithm is so far tested on cartpole, I'll be adding some minor changes soon as the ensuing video is going to be applying it on Atari game.

to train on cartpole: 
---
python main.py --train --cuda --gamma 0.95 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995 --lr 0.001 --buffer_size 2000

to test on cartpole:
---
python main.py --test --cuda

Results on Cartpole:
