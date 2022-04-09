from ludo.ludo import make
import numpy as np
import time
from Players.RandomPlayer import RandomPlayer
from Players.LoadedQPlayer import LoadedQPlayer
import matplotlib.pyplot as plt

AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -10
progress = []
num_wins = 0
# reset the game
num_episodes = 1000
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
q_player = LoadedQPlayer()
player3 = RandomPlayer()
player4 = RandomPlayer()
# create a list of players
players = [player1, q_player, player3, player4]
start = time.time()
for episode in range(1, num_episodes + 1):
    if episode % 100 == 0:
        print("win per", episode, ":", num_wins)

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
        # render for graphical representation of game state
        # env.render()
    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]
    # print(f'Player {players[winner].index} has won!', obs)
    if players[winner] == q_player:
        num_wins += 1

    if episode % 20 == 0:
        progress.append(num_wins/episode)

print(num_wins)
plt.plot(progress)
plt.savefig('progress of test env.')
