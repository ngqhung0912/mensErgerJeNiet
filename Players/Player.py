import numpy as np
from collections import defaultdict


class Player:
    """
    This class serves as an abstract class for all type of player.
    """

    def __init__(self):
        """
        previous_moved_pawn_index: index of previously moved pawn.
        Initialized attributes:
            index: index of current pawn.
            current_reward: reward after current move.
            reward_list: list of rewards after all moves.
            dice: the current dice.
            previous_obs: the previous state.
            relative_position: the current state.
            kicked: does this player get kicked?
            agent: If this player has an agent engine (i.e., the DQN agent).
            killed: The number of times this player kicked other players.
            being_killed: the number of times this player has been kicked by others.
            previous_pos: Previous position of this player's pawn.
            previous action: Previous action taken.
            losses:
            accuracy_list:
        """
        self.previous_moved_pawn_index = None
        self.index = None
        self.current_reward = 0
        self.reward_list = []
        self.dice = None
        self.previous_obs = None
        self.relative_position = None
        self.kicked = False
        self.agent = None
        self.killed = 0
        self.being_killed = 0
        self.previous_pos = [0, 0, 0, 0]
        self.previous_action = None
        self.losses = []
        self.accuracy_list = []

    def load_model(self):
        pass

    def calculate_relative_position(self, obs: list):
        """
        Calculates the relative position of opponents' pawns, with regard to the current player's pawns.
        @param obs: Observation.
        @output: An array with 20 values, include:
                0-4: The current player's pawns' position.
                5-20: Two closest pawns behind within 12 steps, and two closest pawns in front within range 6.
        """

        player_obs = []
        relative_position = []
        for player_elem in obs[self.index]:
            forward_pos = []
            backward_pos = []
            for i in range(4):  # loop over all other player's pawns
                if i == self.index:
                    continue
                if player_elem == 0 or player_elem > 40:
                    continue
                else:
                    relative_factor = i - self.index
                    for enemy_elem in obs[i]:  # loop over each pawn of other player.
                        if enemy_elem == 0 or enemy_elem > 40:
                            # ignore it if the pawn is in starting or in homebase.
                            continue
                        enemy_elem = enemy_elem + 10 * relative_factor  # calculate enemy's relative position
                        if enemy_elem > 40:
                            enemy_elem -= 40
                        distance = (enemy_elem - player_elem)  # calculate relative distance
                        # store two closest behind and two closest in front if it is in range of 12.
                        if 0 < distance < 12:
                            forward_pos.append(distance)
                        elif -12 < distance < 0:
                            backward_pos.append(distance)
            backward_pos.sort()
            forward_pos.sort()
            while len(backward_pos) > 2:
                backward_pos.pop(-1)
            while len(forward_pos) > 2:
                forward_pos.pop(0)
            while len(backward_pos) < 2:
                backward_pos.append(0)
            while len(forward_pos) < 2:
                forward_pos.insert(0, 0)
            # if there is nothing in range of 12, add 0 instead.
            each_relative_position = (backward_pos + forward_pos)
            relative_position.append(each_relative_position)
            player_obs.append(player_elem)
        relative_position.insert(0, player_obs)
        return relative_position

    def handle_move(self, obs: list, info: dict):
        pass

    def handle_nn_input(self, pos: list):
        """
        Normalize nn input and add the dice value.
        @param pos: the current state.
        @return nn_input: Normalized pos.
        """

        for i in range(len(pos)):
            for j in range(len(pos[i])):
                if i == 0:
                    pos[i][j] = pos[i][j] / 44
                else:
                    pos[i][j] = pos[i][j] / 12
        pos = np.array(pos).reshape((20, 1))

        nn_input = np.append(pos, self.dice / 6).reshape((21, 1))
        return nn_input

    def save_previous_obs(self, obs: list):
        """
        Save the previous observation.
        @param obs: previous observation.
        """
        self.previous_obs = obs

    def update_memory(self, reward, moved_pawn_index, done):
        """
        update the replay memory.
        @param reward: Reward after current move
        @param moved_pawn_index: which pawn is being moved this turn?
        @param done: Is the game finished?
        """
        if self.previous_obs is None or len(self.previous_obs) == 0:
            return

        prev_relative_pos = self.calculate_relative_position(self.previous_obs)
        prev_nn_input = self.handle_nn_input(prev_relative_pos)
        current_nn_input = self.handle_nn_input(self.relative_position)
        self.agent.update_replay_memory([
            prev_nn_input,
            moved_pawn_index,
            reward,
            current_nn_input,
            done
        ])

        return

    def handle_reward(self, obs: list):
        """
        Calculates the reward the player received.
        @param obs: Current (raw) observation return from the game engine.
        """

        reward = 0

        if self.index is None or self.previous_obs is None or len(self.previous_obs) == 0:
            return 0
        self.relative_position = Player.calculate_relative_position(self, obs)
        current_obs = obs[self.index]

        """ 
        A very bad take, except value error! I don't remember why, honestly. should have been documented. 
        """
        try:
            if max(filter(lambda x: x < 40, current_obs)) > max(filter(lambda x: x < 40, self.previous_pos)):
                # reward for moving closer to homebase
                reward += 0.1
        except ValueError:
            pass

        for i in range(len(self.previous_pos)):
            if self.previous_pos[i] == 0 and current_obs[i] != 0:  # reward for releasing a piece
                reward += 0.25

        previous_relative_position = Player.calculate_relative_position(self, self.previous_obs)
        previous_relative_position.pop(0)
        previous_relative_position = np.array(previous_relative_position).reshape((16, 1))

        current_relative_position = self.relative_position.copy()
        current_relative_position.pop(0)
        current_relative_position = np.array(current_relative_position).reshape((16, 1))
        for i in range(previous_relative_position.shape[0]):
            if 0 > previous_relative_position[i] > -6 > current_relative_position[i] != 0:
                reward += 0.250  # reward for running away

        if self.kicked:  # punish for being knocked.
            reward -= 0.25
            self.being_killed += 1
            self.kicked = False

        for i in range(len(obs)):  # calculate if an enemy's piece is knocked
            if i == self.index:
                continue
            else:
                for j in range(len(obs[i])):
                    if self.previous_obs[i][j] > 0 and obs[i][j] == 0:
                        reward += 0.25  # knocked a piece
                        self.killed += 1
        return reward

    def save_previous_pos(self, pos: list):
        """
        Save previous position.
        @param pos: Previous position.
        """

        self.previous_pos = pos

    def inform_kicked(self):
        """
        Used by other to inform this player has been kicked, Since this is a turn-based game and this player only
        got activated every time it's their turn.
        """
        self.kicked = True

    def save_action(self, action: list, moved_pawn_index):
        """
        Save the action taken regarding the current move.
        @param action: action taken
        @param moved_pawn_index: Which pawn out of 4 pawns has been moved?
        """
        self.previous_action = action
        self.previous_moved_pawn_index = moved_pawn_index

    def get_action(self):
        """
        Return the previous action.
        """
        return self.previous_action

    def update_tensorboard_stats(self, episode_rewards_list: list, win_rate: float, aggregate_stats):
        """
        Update the tensorboard's stats.
        @param episode_rewards_list: The reward of the current training episodes.
        @param win_rate: Win rate of current training episodes:
        @param aggregate_stats:
        """

        average_reward = sum(episode_rewards_list[-aggregate_stats:]) / len(
            episode_rewards_list[-aggregate_stats:])
        min_reward = min(episode_rewards_list[-aggregate_stats:])
        max_reward = max(episode_rewards_list[-aggregate_stats:])
        if len(self.losses) != 0:
            loss = sum(self.losses) / len(self.losses)
            accuracy = sum(self.accuracy_list) / len(self.accuracy_list)
        else:
            loss = None
            accuracy = None
        self.agent.tensorboard.update_stats(reward_avg=average_reward,
                                            reward_min=min_reward,
                                            reward_max=max_reward,
                                            win_rate=win_rate,
                                            avg_being_knocked=self.being_killed / aggregate_stats,
                                            avg_killed=self.killed / aggregate_stats,
                                            loss=loss,
                                            accuracy=accuracy)

        self.killed = 0
        self.being_killed = 0
        self.losses = []
        self.accuracy_list = []

    def update_model_info(self):
        """
        Update model info when a game is finished.
        """
        model_info = self.agent.get_model_info()
        if self.agent is not None and model_info is not None:
            losses_and_accuracy = model_info.history
            self.accuracy_list.append(losses_and_accuracy['accuracy'][0])
            self.losses.append(losses_and_accuracy['loss'][0])
