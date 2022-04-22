import numpy as np
from collections import defaultdict


class Player:
    def __init__(self):
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
        self.previous_obs = obs

    def update_memory(self, reward, moved_pawn_index, done):
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
        reward = 0

        if self.index is None or self.previous_obs is None or len(self.previous_obs) == 0:
            return 0
        self.relative_position = Player.calculate_relative_position(self, obs)
        current_obs = obs[self.index]
        try:
            if max(filter(lambda x: x < 40, current_obs)) > max(filter(lambda x: x < 40, self.previous_pos)):
                # reward to moving closer to homebase                # reward to moving closer to homebase
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
                reward += 0.250

        if self.kicked:
            reward -= 0.25
            self.being_killed += 1
            self.kicked = False

        for i in range(len(obs)):  # calculate if a piece is knocked
            if i == self.index:
                continue
            else:
                for j in range(len(obs[i])):
                    if self.previous_obs[i][j] > 0 and obs[i][j] == 0:
                        reward += 0.25  # knocked a piece
                        self.killed += 1
        return reward

    def save_previous_pos(self, pos: list):
        self.previous_pos = pos

    def inform_kicked(self):
        self.kicked = True

    def save_action(self, action: list, moved_pawn_index):
        self.previous_action = action
        self.previous_moved_pawn_index = moved_pawn_index

    def get_action(self):
        return self.previous_action

    def update_tensorboard_stats(self, episode_rewards_list: list, win_rate: float, aggregate_stats):
        average_reward = sum(episode_rewards_list[-aggregate_stats:]) / len(
            episode_rewards_list[-aggregate_stats:])
        min_reward = min(episode_rewards_list[-aggregate_stats:])
        max_reward = max(episode_rewards_list[-aggregate_stats:])
        if len(self.losses) != 0:
            loss = sum(self.losses)/len(self.losses)
            accuracy = sum(self.accuracy_list)/len(self.accuracy_list)
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
        model_info = self.agent.get_model_info()
        if self.agent is not None and model_info is not None:
            losses_and_accuracy = model_info.history
            self.accuracy_list.append(losses_and_accuracy['accuracy'][0])
            self.losses.append(losses_and_accuracy['loss'][0])

