import numpy as np
import tensorflow as tf
from LoadedQ.BasePlayer import BasePlayer

global player_example


class LoadedPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.loaded_model = tf.keras.models. \
            load_model('models/ludo__model_1500__21-100-100-50-50-4__0.262.model')
        self.relative_position = None

    def handle_move(self, obs: list, info: dict):
        self.index = info['player']
        self.dice = info['eyes']
        self.relative_position = BasePlayer.calculate_relative_position(self, obs)
        nn_input = self.handle_nn_input(self.relative_position)
        nn_output = self.loaded_model.predict(np.array(nn_input).reshape(1, 21)).reshape((4,))
        nn_output = np.ndarray.tolist(nn_output)
        return nn_output



def player(obs, info):
    """
    defines a random player: returns a random action as a (4,) numpy array
regardless the game state
    """
    # here goes your code
    # do not load your model here but use the main() function icm with a global variable
    # action  = np.random.random_sample(size = 4)
    return player_example.handle_move(obs, info)


# any other code that should run during import define in function main()
def main():
    # do all required initialisation here
    # use relative paths for access to stored files that you require
    # use global variables to make sure the player() function has access
    global player_example
    player_example = LoadedPlayer()
    pass

main()