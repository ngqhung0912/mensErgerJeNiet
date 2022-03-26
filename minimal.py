from ludo.ludo import make
import numpy as np


def random_player():
    """
        defines a random player: returns a random action regardless the gamestate
    """
    return np.random.random_sample(size=4)


winner_list = []
for _ in range(1000):
    # create an instance of the game with 4 players
    env = make(num_players=4)

    # reset the game
    obs, rew, done, info = env.reset()

    while True:
        # get an action from the random player
        action = random_player()

        # pass the action and get the new game state
        obs, rew, done, info = env.step(action)

        # render for graphical representation of game state
        # env.render()

        # quit if game is finished
        if done:
            winner_list.append(info['player'])
            break

