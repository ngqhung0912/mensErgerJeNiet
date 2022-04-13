from model.ModifiedTensorBoard import ModifiedTensorBoard as mtb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import random
import tensorflow as tf

class Agent:
    def __init__(self, model_name: str, discount_rate: float, learning_rate: float, episodes: int):

        self.replay_memory_size = 100000
        self.min_replay_memory_size = 100
        self.minibatch_size = 64
        self.update_target_range = 5
        self.update_logs = 1000
        self.model_name = model_name
        self.discount_rate = discount_rate
        self.input_shape = 21
        self.output_shape = 4
        self.learning_rate = learning_rate
        self.log_num = 0

        # Main model
        # self.model = tf.keras.models.\
        #     load_model('/Users/hungnguyen/mensErgerJeNiet/models/ludo__model_1500__21-100-100-50-50-4__0.3.model')
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        # self.target_model = tf.keras.models.\
        #     load_model('/Users/hungnguyen/mensErgerJeNiet/models/ludo__target_model_1500__21-100-100-50-50-4__0.3.model')

        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.replay_memory_size)

        # Custom tensorboard object
        self.tensorboard = mtb(log_dir="logs/lr={}-dr={}-num_eps={}".format(learning_rate, discount_rate, episodes))

        # Used to count when to update target network xwith main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Dense(100, input_dim=21, kernel_initializer='normal',
                        activation="relu"))

        model.add(Dense(50, input_dim=100,
                        kernel_initializer="normal", activation="relu"))

        model.add(Dense(100, input_dim=100,
                        kernel_initializer="normal", activation="relu"))

        model.add(Dense(4, input_dim=100,
                        kernel_initializer="normal", activation="softmax"))
        model.compile(loss="mse",
                      optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mae'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition: list):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]).reshape(self.minibach_size, 21)  # normalize
        current_qs_list = self.model.predict(current_states, batch_size=self.minibatch_size)
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]).reshape(self.minibatch_size, 21)   # normalize
        future_qs_list = self.target_model.predict(new_current_states)
        input_batch = []
        output_batch = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount_rate * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            # current_qs = current_qs_list[index]
            # current_qs[action] = new_q
            current_qs_list[index] = new_q

            # And append to our training data
            input_batch.append(current_state)
            output_batch.append(current_qs_list[index])

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(input_batch).reshape(self.minibatch_size, 21),
                       np.array(output_batch).reshape(self.minibatch_size, 4),
                       batch_size=self.minibatch_size,
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard] if self.log_num == self.update_logs else None)


        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
            self.log_num += 1

        if self.log_num == self.update_logs:
            self.log_num = 0

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.update_target_range:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state: list):
        state = np.array(state).reshape((1, 21))
        return self.model.predict(state)

    def save_model(self, progress):
        self.model.save(
            'models/ludo__model_1500__21-100-100-50-50-4__{}.model'.format(progress))

        self.target_model.save(
            'models/ludo__target_model_1500__21-100-100-50-50-4__{}__.model'.format(progress))
