# import required packages
from ludo.ludo import make
import numpy as np
import os

# import players
from Players.PlayerExample import player as player1
from ludo.ludo import random_player
from Players.QPlayer import player as QPlayer

# constants
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']

# create a list of players
players = [player1, random_player, QPlayer, random_player ]

# create an instance of the game with 4 players
env = make(num_players=4)

# reset the game
obs, rew, done, info = env.reset()

# play the game until finished
while True:
    # get an action from the current player
    current_player = players[info['player']]
    action = current_player(obs, info)

    # pass the action and get the new gamestate
    obs, rew, done, info = env.step(action)

    # render for graphical representation of gamestate
    env.render()

    # quit if game is finished
    if done:
        break


# compute the winner / ranking
scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
winner = np.argsort(scores)[-1]
print(f'Player {PLAYER2COLOR[winner]} has won!')

input('Press any key to close')