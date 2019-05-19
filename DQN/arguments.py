import argparse

def get_args():
    parse= argparse.ArgumentParser()
    # network related arguments
    parse.add_argument("--env_name", type=str, default="CartPole-v0", help="The environment")
    parse.add_argument("--batch_size", type= int, default= 128, help="The batch size of the data we train from")
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate of the algorithm')
    parse.add_argument('--seed', type=int, default= 123, help='the seed value to initialise the pseudo random generator at the same random values')
    parse.add_argument("--train", action= 'store_true', help="Train the model only")
    parse.add_argument("--test", action= 'store_true', help="Test the model only (must load a model!)")
    parse.add_argument("--save_path", type= str, default= r'/home/neurobeast/Documents/SamTukra/DRL/DQN/models/', help="The path where you want to save your trained model and for loading the same model when testing")
    parse.add_argument("--load_model", type= str, default= r'/home/neurobeast/Documents/SamTukra/DRL/DQN/models/_CartPole-v0_dqn_10000episodes.pt', help="The path where you want to save your trained model and for loading the same model when testing")
    parse.add_argument("--loss_type", type= str, default= 'mse', help="The loss type you want to update the model with")
    # algorithm related arguments
    parse.add_argument('--gamma', type=float, default=0.95, help='the discount factor i.e. gamma')
    parse.add_argument('--tau', type=float, default=1e-3, help='parameter for performing soft update on target net')
    parse.add_argument("--buffer_size", type=int, default=10000, help="max memory length for the Replay Buffer")
    parse.add_argument("--epsilon_start", type=float, default=1.0, help="starting epsilon")
    parse.add_argument("--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parse.add_argument("--epsilon_decay", type=float, default=0.99, help="epsilon decay parameter")
    # implementation related arguments
    parse.add_argument("--max_TimeSteps", type=int, default=1000, help="max time steps 1 episodes last! if terminal state is not reached the env will reset after this")
    parse.add_argument("--target_net_update", type=int, default=10, help="The environment you want the agent to perform in")
    parse.add_argument('--learning-starts', type=int, default=10, help='the frames start to learn')
    parse.add_argument("--learn_frequency", type=int, default=4, help=" no.of time steps you want to print stuff in")
    parse.add_argument("--display_interval", type=int, default=100, help=" no.of time steps you want to print stuff in")
    parse.add_argument("--episodes", type=int, default=70000, help="The total no of episodes I want to train the model for")
    parse.add_argument("--iter", type=int, default=1e7, help="The total no of iterations I want to train the model for")
    parse.add_argument("--cuda", action= 'store_true', help="use GPU for training else, CPU")

    args= parse.parse_args()

    return args
# CartPole-v0 
    