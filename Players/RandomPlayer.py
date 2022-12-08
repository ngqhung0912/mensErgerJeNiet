from Players.Player import Player
import numpy as np

global random_player


class RandomPlayer(Player):
    def __init__(self):
        super(RandomPlayer, self).__init__()

    def handle_move(self, obs: list, info: dict):
        self.index = info['player']
        self.dice = info['eyes']
        # Player.generate_nn_input(relative_position)
        return np.random.random_sample(size=4)


def player(obs, info):
    """
    defines a random player: returns a random action as a (4,) numpy array
    regardless the game state
    """
    # here goes your code
    # do not load your model here but use the main() function icm with a global variable
    return random_player.handle_move(obs, info)


# any other code that should run during import define in function main()
def main():
    # do all required initialisation here
    # use relative paths for access to stored files that you require
    # use global variables to make sure the player() function has access
    global random_player
    random_player = RandomPlayer()
    pass


main()
