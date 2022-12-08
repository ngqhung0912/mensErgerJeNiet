import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import tensorflow as tf
import pickle
from os import path


class Agent:
    """
    This class serves as the Deep Q network agent. It contains the models and the replay memory.
    """

    def __init__(self, model_name: str, discount_rate: float, learning_rate: float, tensorboard,
                 training: bool, model_dir=None):
        """
        initialization of the Agent class, which is the core engine of the Deep Q Learning player.
        @param discount_rate: Discounting factor of future rewards.
        @param learning_rate: learning rate of the model.x
        @param tensorboard: Tensorboard to log different hyperparameters.
        @param training: True if this is a training run, false otherwise.
        @param model_dir: if we want to load a pre-trained model in.
        replay_memory_size:  How many 'experience' should the model have at the same time, i.e., at some point it will
        ditch some learned knowledge to make space for new one.
        min_replay_memory_size: minimum replay memory size to start sample minibatch.
        minibatch_size: How many states got sampled from the replay memory size to train each time.
        update_target_range:  How often the weight of the model got transfer to the main model's.
        update_logs: How often the tensorboard got updated.
        input_shape: Input layer's shape.
        output_shape: output layer's shape.
        optimizer: The optimizer method.
        """
        self.replay_memory_size = 200000
        self.min_replay_memory_size = 100
        self.minibatch_size = 32
        self.update_target_range = 5
        self.update_logs = 1000
        self.model_name = model_name
        self.discount_rate = discount_rate
        self.input_shape = 21
        self.output_shape = 4
        self.learning_rate = learning_rate
        self.log_num = 0
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # Main model
        if training:
            self.model = self.create_model()
        else:
            self.model = tf.keras.models. \
                load_model(model_dir)

        # Target network
        self.target_model = self.create_model()

        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        if path.exists('experience/experience.txt'):
            with open('experience/experience.txt', 'rb') as pickle_file:
                self.replay_memory = pickle.load(pickle_file)
        else:
            self.replay_memory = deque(maxlen=self.replay_memory_size)

        # Custom tensorboard object
        self.tensorboard = tensorboard

        # Used to count when to update target network x with main network's weights
        self.target_update_counter = 0
        self.model_info = None

    def create_model(self):
        """
        Create the Deep Learning model. Current architecture:
        L1: 21x16,  RelU activation
        L2: 16x16, ReLU activation
        L3: 16x16, ReLU activation
        L4: 16x16, ReLU activation
        L5: 16x4,  Softmax activation
        Loss Function: Categorical CrossEntropy
        Metrics: Accuracy
        Optimizer: Adam
        """
        model = Sequential()

        model.add(Dense(16, input_dim=21, activation="relu"))

        model.add(Dense(16, input_dim=16, activation="relu"))
        model.add(Dense(16, input_dim=16, activation="relu"))
        model.add(Dense(16, input_dim=16, activation="relu"))

        model.add(Dense(4, input_dim=16,
                        kernel_initializer="normal", activation="softmax")
                  )
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition: list):
        """
        Add step's data to a memory replay array.
        @param transition: An array consists of:
            [prev_nn_input: previous state,
            moved_pawn_index: current state,
            reward: reward earned,
            current_nn_input: current state,
            done: if the game is finished]
        """
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        """
        Train the main network every step during episode.
        @param terminal_state: If the game is done or not.
        """

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]).reshape(self.minibatch_size, 21)

        current_qs_list = self.model.predict(current_states, batch_size=self.minibatch_size)
        # Get future states from minibatch, then query NN model for Q values

        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]).reshape(self.minibatch_size,
                                                                                           21)  # normalize
        future_qs_list = self.target_model.predict(new_current_states)
        input_batch = []
        output_batch = []

        # enumerate over batches
        for index, (current_state, moved_pawn_index, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.discount_rate * max_future_q

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[moved_pawn_index] = new_q

            # And append to our training data
            input_batch.append(current_state)
            output_batch.append(current_qs)

        # Fit on all samples as one batch, log only on specified state
        self.model_info = self.model.fit(np.array(input_batch).reshape(self.minibatch_size, 21),
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

    def get_qs(self, state: list):
        """
        Queries main network for Q values given current observation space (environment state)
        @param state: current state.
        """
        state = np.array(state).reshape((1, 21))
        action = self.model.predict(state)
        return action

    def save_model(self, model_name):
        """
        Save the trained model, using pickle.
        @param model_name: model's name.
        """
        model_name = 'models/ludo' + model_name + '.model'
        self.model.save(model_name)
        with open("experience/experience.txt", "wb") as file:
            pickle.dump(self.replay_memory, file)

    def get_model_info(self):
        """
        Get the model's info, return from the model.fit function.
        """
        return self.model_info
