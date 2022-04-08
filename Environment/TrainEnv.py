from ludo.ludo import make
import numpy as np
import matplotlib.pyplot as plt
import time
from Players.RandomPlayer import RandomPlayer
from Players.QPlayer import QPlayer
from Players.LoadedQPlayer import LoadedQPlayer


final_reward_list = []
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -10
episode_rewards_list = []
model_name = 'ludo'
progress = []
num_wins = 0
# reset the game
num_episodes = 100
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
training_player = QPlayer(model_name, epsilon=1)
player3 = RandomPlayer()
player4 = RandomPlayer()
# create a list of players
players = [player1, training_player, player3, player4]

start_time = time.time()
for episode in range(1, num_episodes + 1):
    if episode % 100 == 0:
        end_10_eps_time = time.time()
        print("time per 100 games:", end_10_eps_time - start_time)
        print('win per 100 games:', num_wins)

        start_time = end_10_eps_time

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
            if not done:
                training_player.update_memory(action, reward, done)
                training_player.agent.train(done)

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
        num_wins += 1

    episode_reward += final_reward
    episode_rewards_list.append(episode_reward)
    training_player.update_memory(action, reward, done)
    training_player.agent.train(done)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(episode_rewards_list[-AGGREGATE_STATS_EVERY:]) / len(
            episode_rewards_list[-AGGREGATE_STATS_EVERY:])
        min_reward = min(episode_rewards_list[-AGGREGATE_STATS_EVERY:])
        max_reward = max(episode_rewards_list[-AGGREGATE_STATS_EVERY:])
        training_player.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                       reward_max=max_reward, epsilon=training_player.epsilon)
        # Save model, but only when min reward is greater or equal a set value
    training_player.handle_endgame()
    progress.append(num_wins/episode)
plt.figure(1)
plt.plot(episode_rewards_list)
plt.savefig('rewards.png')

print(num_wins)


plt.figure(2)
plt.plot(progress)
plt.savefig('training_progress.png')

training_player.agent.save_model(progress[-1])


