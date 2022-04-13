# import required packages
from ludo.ludo import make
import numpy as np
import os
import matplotlib.pyplot as plt
# import players
from ludo.ludo import random_player as player1
from ludo.ludo import random_player as rd1
from ludo.ludo import random_player
from LoadedQ.LoadedPlayer import player as player2
# constants
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']

# create a list of players
players = [player1, player2, random_player, random_player]

# create an instance of the game with 4 players

# reset the game
reward_list = []
# play the game until finished
for _ in range(1):
    reward = 0

    env = make(num_players=4)
    obs, rew, done, info = env.reset()

    while True:
        # get an action from the current player
        current_player = players[info['player']]
        action = current_player(obs, info)

        # pass the action and get the new gamestate
        obs, rew, done, info = env.step(action)
        if current_player == rd1:
            reward += rew

        # render for graphical representation of gamestate
        # env.render()

        # quit if game is finished
        if done:
            break


    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]
    # print(f'Player {PLAYER2COLOR[winner]} has won!')

    if players[winner] != rd1:
        reward -= 1
    else:
        reward += 1

    reward_list.append(reward)


