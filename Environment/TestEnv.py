from ludo.ludo import make
import numpy as np
import time
from Players.RandomPlayer import RandomPlayer
from Players.LoadedQPlayer import LoadedQPlayer
from EnvFunctions import EnvFunctions



final_reward_list = []
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -2
episode_rewards_list = []
model_name = 'ludo'
progress = []
num_wins = 0
# reset the game
num_episodes = 500
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
fun = EnvFunctions()

learned_player = LoadedQPlayer(model_name, episodes=num_episodes)
strategy_player = RandomPlayer()
player4 = RandomPlayer()
# create a list of players
players = [player1, learned_player, strategy_player, player4]

start_time = time.time()
for episode in range(1, num_episodes + 1):

    learned_player.agent.tensorboard.step = episode
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
        if learned_player.index is not None:
            learned_player.save_previous_pos(obs[learned_player.index])

        action = current_player.handle_move(obs, info)
        # pass the action and get the new game state
        learned_player.save_previous_obs(obs)
        obs, reward, done, info, moved_pawn_index = env.step(action)
        if learned_player.index is not None \
                and obs[learned_player.index] != learned_player.previous_pos \
                and info['player'] != learned_player.index \
                and info['player'] != learned_player.index + 1:
            learned_player.inform_kicked()

        # handle the reward
        if isinstance(current_player, LoadedQPlayer):
            move_reward = current_player.handle_reward(obs)
            episode_reward += move_reward
            if not done and moved_pawn_index != -1:
                learned_player.agent.train(done)
                learned_player.update_memory(move_reward, moved_pawn_index, done)
                learned_player.update_model_info()
            learned_player.save_action(action, moved_pawn_index)

        # render for graphical representation of game state
    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]

    # print(f'Player {players[winner].index} has won!', obs)
    if players[winner] != learned_player:
        final_reward = -1
    else:
        final_reward = 5
        num_wins += 1

    episode_reward += final_reward
    episode_rewards_list.append(episode_reward)
    learned_player.update_memory(episode_reward, np.argmax(learned_player.get_action()), done)
    learned_player.agent.train(done)
    learned_player.update_model_info()

    if episode % AGGREGATE_STATS_EVERY == 0 and episode != 1:
        win_rate = num_wins / AGGREGATE_STATS_EVERY
        end_time = time.time()
        print('win rate after {} games: {}'.format(episode, win_rate))
        learned_player.update_tensorboard_stats(episode_rewards_list, win_rate, AGGREGATE_STATS_EVERY)

        num_wins = 0
        progress.append(win_rate)

    learned_player.handle_endgame()


learned_player.agent.save_model('testrun_2')
