from ludo.ludo import make
import numpy as np
import os

# import players
from ludo.ludo import random_player
from Players.RandomPlayer import RandomPlayer

# constants
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
player2 = RandomPlayer()
player3 = RandomPlayer()
player4 = RandomPlayer()

# create a list of players
players = [player1, player2, player3, player4]

# create an instance of the game with 4 players
env = make(num_players=4)

# reset the game
obs, reward, done, info = env.reset()

# play the game until finished
while True:
    # get an action from the current player
    current_player = players[info['player']]

    action = current_player.handle_move(obs, info)
    # pass the action and get the new game state
    obs, reward, done, info = env.step(action)
    # handle the reward
    current_player.handle_reward(obs, reward)
    # render for graphical representation of game state
    env.render()

    # quit if game is finished
    if done:
        break

# compute the winner / ranking
scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
winner = np.argsort(scores)[-1]
print(f'Player {PLAYER2COLOR[winner]} has won!')

input('Press any key to close')
