from ludo.ludo import make
import numpy as np
import time
from Players.RandomPlayer import RandomPlayer
from Players.LoadedQPlayer import LoadedQPlayer
import matplotlib.pyplot as plt
from Players.StrategyPlayer import StrategyPlayer
from EnvFunctions import EnvFunctions

AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -10
q_progress = []
strategy_progress = []
# reset the game
num_episodes = 1000
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']
player1 = RandomPlayer()
q_player = LoadedQPlayer()
strategy_player = StrategyPlayer()
player4 = RandomPlayer()
# create a list of players
players = [player1, q_player, strategy_player, player4]
start = time.time()
strategy_win = 0
load_q_win = 0
fun = EnvFunctions()

for episode in range(1, num_episodes + 1):
    if episode % 100 == 0:
        print(episode, load_q_win, strategy_win)

    # create an instance of the game with 4 players
    env = make(num_players=4)

    obs, reward, done, info = env.reset()

    # play the game until finished
    while not done:
        # get an action from the current player
        current_player = players[info['player']]
        print(current_player)
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
        load_q_win += 1
    elif players[winner] == strategy_player:
        strategy_win += 1

    if episode > 20:
        q_progress.append(load_q_win/episode)
        strategy_progress.append(strategy_win/episode)

fun.plot_array(strategy_progress, "strategy progress", x_label="number of games", y_label="win rate")
fun.plot_array(q_progress, "Q progress", x_label="number of games", y_label="win rate")

