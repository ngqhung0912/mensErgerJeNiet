from ludo.ludo import make
import numpy as np
import time
from Players.RandomPlayer import RandomPlayer
from Players.QPlayer import QPlayer
from EnvFunctions import EnvFunctions
from Players.StrategyPlayer import StrategyPlayer


final_reward_list = []
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -10
episode_rewards_list = []
model_name = 'ludo'
progress = []
num_wins = 0
# reset the game
num_episodes = 10
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
fun = EnvFunctions()

training_player = QPlayer(model_name, epsilon=1, episodes=num_episodes)
strategy_player = StrategyPlayer()
player4 = RandomPlayer()
# create a list of players
players = [player1, training_player, strategy_player, player4]


start_time = time.time()
for episode in range(1, num_episodes + 1):

    training_player.agent.tensorboard.step = episode
    episode_reward = 0
    step = 1

    # create an instance of the game with 4 players
    env = make(num_players=4)

    obs, reward, done, info = env.reset()

    # play the game until finished
    while not done:
        # get an action from the current player
        current_player = players[info['player']]
        action = current_player.handle_move(obs, info)
        # pass the action and get the new game state
        obs, reward, done, info = env.step(action)
        # handle the reward

        if isinstance(current_player, QPlayer):
            episode_reward += current_player.handle_reward(obs, info['player'])
            if not done:
                training_player.update_memory(action, reward, done)
                training_player.agent.train(done)

        training_player.save_previous_obs(obs)

        # render for graphical representation of game state
        env.render()
    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]
    # print(f'Player {players[winner].index} has won!', obs)
    if players[winner] != training_player:
        final_reward = -1
    else:
        final_reward = 1
        num_wins += 1

    episode_reward += final_reward
    episode_rewards_list.append(episode_reward)
    training_player.update_memory(action, reward, done)
    training_player.agent.train(done)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        win_rate = num_wins / episode
        end_time = time.time()
        print('time per {} games: {}'.format(episode, end_time - start_time))
        print('win rate per {} games: {}'.format(episode, win_rate))
        training_player.update_tensorboard_stats(episode_rewards_list, win_rate, AGGREGATE_STATS_EVERY,
                                                 time=(end_time - start_time))

    training_player.handle_endgame()
    progress.append(num_wins / episode)


fun.plot_array(progress, "training progress", x_label="number of games", y_label="win rate",
               additional_infos=training_player.info_array)
