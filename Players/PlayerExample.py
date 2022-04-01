# import required packages
import numpy as np
from Players.Player import Player


class PlayerExample(Player):
    def __init__(self):
        super().__init__()
        self.shit = [0, 0, 1, 0]

    def load_model(self):
        return self.shit
    pass


def player(obs, info):
    """
    defines a random player: returns a random action as a (4,) numpy array
regardless the game state
    """
    # here goes your code
    # do not load your model here but use the main() function icm with a global variable
    # action  = np.random.random_sample(size = 4)
    return action


# any other code that should run during import define in function main()
def main():
    # do all required initialisation here
    # use relative paths for access to stored files that you require
    # use global variables to make sure the player() function has access
    player_example = PlayerExample()
    global action
    action = player_example.load_model()
    pass


main()
