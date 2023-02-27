import math
import time
from environment import *
from state import State
import numpy as np


class RLAgent:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.learning_rate = environment.alpha  # alpha
        self.states = [environment.get_init_state()]
        self.states_set = {environment.get_init_state()}
        self.q_table = np.zeros((len(self.states), len(ROBOT_ACTIONS)))

        # epsilon
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 1750

        # 0.01, 1750

        self.frame_idx = 0
        self.total_rewards = 0
        self.rewards = []
        self.episode_no = 0

        self.exploit_prob = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1.0 * self.frame_idx / self.epsilon_decay)

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        #
        # TODO: Implement your Q-learning training loop here.
        #
        # threshold = self.environment.evaluation_reward_tgt

        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 2000
        rewards = []
        r100 = -math.inf

        threshold = self.environment.training_reward_tgt
        threshold2 = self.environment.evaluation_reward_tgt
        while self.environment.get_total_reward() > threshold and r100 < (threshold2 * 1.28):
            # perform one episode
            state = self.environment.get_init_state()
            episode_reward = 0

            while not self.environment.is_solved(state):
                self.exploit_prob = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1.0 * self.frame_idx / self.epsilon_decay)

                state_index = self.states.index(state)
                action = self.choose_action(state_index)
                reward, next_state = self.environment.perform_action(state, action)
                episode_reward += reward

                # add new state to state list and q table
                if next_state not in self.states_set:
                    self.states.append(next_state)
                    self.states_set.add(next_state)
                    new_q_table_entry = np.zeros((1, len(ROBOT_ACTIONS)))
                    self.q_table = np.vstack([self.q_table, new_q_table_entry])

                self.frame_idx += 1

                # update q table
                q_old = self.q_table[state_index, action]

                if self.environment.is_solved(next_state):
                    q_best_next = 0
                else:
                    next_state_index = self.states.index(next_state)
                    q_best_next = np.max(self.q_table[next_state_index])

                target = reward + self.environment.gamma * q_best_next
                q_new = q_old + self.learning_rate * (target - q_old)

                self.q_table[state_index, action] = q_new
                state = next_state

            rewards.append(episode_reward)
            r100 = np.mean(rewards[-100:])
        print(f"R100: {r100}")

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        state_index = self.states.index(state)
        if random.uniform(0, 1) < self.exploit_prob:
            # explore - i.e. choose a random action
            action = random.choice(ROBOT_ACTIONS)
        else:
            action = np.argmax(self.q_table[state_index])
        return action

    # === SARSA ========================================================================================================
    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        #
        init_state = self.environment.get_init_state()
        threshold1 = self.environment.training_reward_tgt
        threshold2 = self.environment.evaluation_reward_tgt

        # start epsilon at this value
        self.exploit_prob = self.epsilon_start
        rewards = []
        max_r100 = -math.inf
        r100 = math.inf
        eval_reward = 0
        r100_avg = -math.inf
        while self.environment.get_total_reward() > threshold1:
            episode_reward = 0
            episode_start = self.frame_idx
            reward = 0
            eval_reward = 0

            # perform one episode
            state = self.environment.get_init_state()
            state_index = self.states.index(state)

            action = self.choose_action(state_index)

            while not self.environment.is_solved(state):
                state_index = self.states.index(state)

                self.exploit_prob = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1.0 * self.frame_idx / self.epsilon_decay)
                reward, next_state = self.environment.perform_action(state, action)
                eval_reward += reward

                self.frame_idx += 1
                self.total_rewards += reward
                episode_reward += reward

                # add new state to state list and q table
                if next_state not in self.states_set:
                    self.states.append(next_state)
                    self.states_set.add(next_state)
                    new_q_table_entry = np.zeros((1, len(ROBOT_ACTIONS)))
                    self.q_table = np.vstack([self.q_table, new_q_table_entry])

                # update q table
                # Q_new(s,a) <-- Q_old(s,a) + alpha * (R + gamma*Q(s',a') - Q_old(s, a))
                # S' == next_state, a' == next_action
                next_state_index = self.states.index(next_state)
                next_action = self.choose_action(next_state_index)

                q_old = self.q_table[state_index, action]
                q_next_old = self.q_table[next_state_index, next_action]

                q_new = q_old + self.learning_rate * (reward + (self.environment.gamma * q_next_old) - q_old)

                self.q_table[state_index, action] = q_new
                state = next_state
                action = next_action

            rewards.append(episode_reward)
            r100 = np.mean(rewards[-100:])
            if len(rewards) > 100:
                r100_avg = r100
            if r100 > max_r100:
                max_r100 = r100
            # print(f"Frame: {self.frame_idx}, Episode {self.episode_no}, steps taken {self.frame_idx - episode_start}, reward: {episode_reward}, R100: {r100}, max R100: {max_r100}, epsilon: {self.exploit_prob}")
            self.episode_no += 1
            # print(f"Total Rewards: {eval_reward}")
            # print(f"Episode {self.episode_no}, steps taken {self.frame_idx - episode_start}, reward: {episode_reward}, R100: {np.mean(self.rewards[-100:])}, epsilon: {self.exploit_prob}")

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        state_index = self.states.index(state)
        if random.uniform(0, 1) < self.exploit_prob:
            # explore - i.e. choose a random action
            action = random.choice(ROBOT_ACTIONS)
        else:
            action = np.argmax(self.q_table[state_index])
        return action

    # === Helper Methods ===============================================================================================
    def choose_action(self, state_index):
        if random.uniform(0, 1) < self.exploit_prob:
            # explore - i.e. choose a random action
            action = random.choice(ROBOT_ACTIONS)
        else:
            action = np.argmax(self.q_table[state_index])
        
        return action
