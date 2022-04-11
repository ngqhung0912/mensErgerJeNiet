from ludo.ludo import make
import numpy as np
import time
from Players.RandomPlayer import RandomPlayer
from Players.QPlayer import QPlayer
from EnvFunctions import EnvFunctions
from Players.StrategyPlayer import StrategyPlayer

final_reward_list = []
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -5
episode_rewards_list = []
model_name = 'ludo'
progress = []
num_wins = 0
# reset the game
num_episodes = 1000
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

    obs, _, done, info = env.reset()

    # play the game until finished
    while not done:
        # get an action from the current player
        # env.render()

        current_player = players[info['player']]
        # if isinstance(current_player, QPlayer):
        if training_player.index is not None:
            training_player.save_previous_pos(obs[training_player.index])

        action = current_player.handle_move(obs, info)
        # pass the action and get the new game state
        training_player.save_previous_obs(obs)
        obs, reward, done, info = env.step(action)
        if training_player.index is not None \
                and obs[training_player.index] != training_player.previous_pos \
                and info['player'] != training_player.index \
                and info['player'] != training_player.index +1:
            training_player.inform_kicked()

        # handle the reward
        if isinstance(current_player, QPlayer):
            move_reward = current_player.handle_reward(obs)
            episode_reward += move_reward
            if not done:
                training_player.update_memory(action, move_reward, done)
                training_player.agent.train(done)
        # render for graphical representation of game state
    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]

    # print(f'Player {players[winner].index} has won!', obs)
    if players[winner] != training_player:
        final_reward = -5
    else:
        final_reward = 5
        num_wins += 1

    episode_reward += final_reward
    episode_rewards_list.append(episode_reward)
    training_player.update_memory(action, episode_reward, done)
    training_player.agent.train(done)

    if episode % AGGREGATE_STATS_EVERY == 0 and episode != 1:
        win_rate = num_wins / episode
        end_time = time.time()
        print('time for {} games: {}'.format(episode, end_time - start_time))
        print('win rate after {} games: {}'.format(episode, win_rate))
        training_player.update_tensorboard_stats(episode_rewards_list, win_rate, AGGREGATE_STATS_EVERY,
                                                 time=(end_time - start_time))

    training_player.handle_endgame()
    progress.append(num_wins / episode)

fun.plot_array(progress, "training progress", x_label="number of games", y_label="win rate",
               additional_infos=training_player.info_array)

training_player.agent.save_model(progress[-1])
