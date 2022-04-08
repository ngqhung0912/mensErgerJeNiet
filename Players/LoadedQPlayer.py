from model.ModifiedTensorBoard import ModifiedTensorBoard as mtb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
from Players.Player import Player


class LoadedQPlayer(Player):
    def __init__(self):
        super().__init__()
        self.loaded_model = tf.keras.models.load_model('/Users/hungnguyen/mensErgerJeNiet/models/ludo_____8.20max____4.20avg____2.15min__1649439603.model')
        self.relative_position = None


    def handle_move(self, obs: list, info: dict):
        self.index = info['player']
        self.dice = info['eyes']
        self.relative_position = Player.calculate_relative_position(self, obs)
        nn_input = self.handle_nn_input(self.relative_position)
        nn_output = self.loaded_model.predict(np.array(nn_input).reshape(1, 21)).reshape((4,))
        nn_output = np.ndarray.tolist(nn_output)

        return nn_output








