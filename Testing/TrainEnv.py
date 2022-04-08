from ludo.ludo import make
import numpy as np
import matplotlib.pyplot as plt
import time
from Players.RandomPlayer import RandomPlayer
from Players.QPlayer import QPlayer

final_reward_list = []
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -10
episode_rewards_list = []
model_name = 'ludo'
# reset the game
num_episodes = 1000
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
training_player = QPlayer(model_name, epsilon=1)
player3 = RandomPlayer()
player4 = RandomPlayer()
# create a list of players
players = [player1, training_player, player3, player4]

for episode in range(1, num_episodes+1):
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
        # if info['eyes'] == 6:
        #     print()

        if isinstance(current_player, QPlayer):
            episode_reward += current_player.handle_reward(obs, info['player'])

        training_player.save_previous_obs(obs)

        # render for graphical representation of game state
        # env.render()
    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]
    # print(f'Player {players[winner].index} has won!', obs)
    if players[winner] != training_player:
        final_reward = -1
    else:
        final_reward = 1

    episode_reward += final_reward
    episode_rewards_list.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(episode_rewards_list[-AGGREGATE_STATS_EVERY:])/len(episode_rewards_list[-AGGREGATE_STATS_EVERY:])
        min_reward = min(episode_rewards_list[-AGGREGATE_STATS_EVERY:])
        max_reward = max(episode_rewards_list[-AGGREGATE_STATS_EVERY:])
        training_player.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=training_player.epsilon)

        # Save model, but only when min reward is greater or equal a set value
        # if min_reward >= MIN_REWARD:
        #     training_player.agent.model.save(f'models/{model_name}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


plt.plot(episode_rewards_list)
plt.savefig('trial.png')


