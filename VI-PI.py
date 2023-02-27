import numpy as np
import constants
from environment import *
from state import State


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.values = {}
        self.policy = {}
        self.states = []
        self.discount = self.environment.gamma
        self.differences = []
        self.converged = False
        self.t_model = None
        self.r_model = None
        self.la_policy = None

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        frontier = [self.environment.get_init_state()]
        actions = constants.ROBOT_ACTIONS
        while len(frontier) > 0:
            state = frontier.pop(0)
            for action in actions:
                # all possible successors

                normal = [action]
                drift_cw = [SPIN_RIGHT, action]
                drift_ccw = [SPIN_LEFT, action]
                double = [action, action]
                drift_cw_double = [SPIN_RIGHT, action, action]
                drift_ccw_double = [SPIN_LEFT, action, action]
                possible_moves = [normal, drift_cw, drift_ccw, double, drift_cw_double, drift_ccw_double]
                for noiseAction in possible_moves:
                    new_state = state
                    for move in noiseAction:
                        reward, new_state = self.environment.apply_dynamics(new_state, move)
                    # check if new state is valid
                    if new_state not in self.values.keys():
                        self.values[new_state] = -40
                        frontier.append(new_state)
        print("states finished initialising")

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.converged

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        new_values = dict()
        new_policy = dict()
        for state in self.values.keys():
            if self.environment.is_solved(state):
                new_values[state] = 0.0
                continue

            action_values = dict()
            for action in constants.ROBOT_ACTIONS:
                total = 0.0
                # apply each possible action
                for possible_action, p in self.stoch_action(action):
                    new_state = state
                    rewards = []
                    # Calculate V(s) for an action
                    for a in possible_action:
                        reward, new_state = self.environment.apply_dynamics(new_state, a)
                        rewards.append(reward)
                    total += p * (min(rewards) + (self.environment.gamma * self.values[new_state]))

                action_values[action] = total
            new_values[state] = max(action_values.values())
            new_policy[state] = self.dict_argmax(action_values)

        differences = [abs(self.values[s] - new_values[s]) for s in self.values.keys()]
        max_diff = max(differences)
        self.differences.append(max_diff)

        if max_diff < self.environment.epsilon:
            self.converged = True

        # Update values
        self.values = new_values
        self.policy = new_policy

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.values.get(state, 0)

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy.get(state, 0)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        frontier = [self.environment.get_init_state()]
        actions = constants.ROBOT_ACTIONS
        while len(frontier) > 0:
            state = frontier.pop(0)
            for action in actions:
                # all possible successors
                normal = [action]
                drift_cw = [SPIN_RIGHT, action]
                drift_ccw = [SPIN_LEFT, action]
                double = [action, action]
                drift_cw_double = [SPIN_RIGHT, action, action]
                drift_ccw_double = [SPIN_LEFT, action, action]
                possible_moves = [normal, drift_cw, drift_ccw, double, drift_cw_double, drift_ccw_double]
                for noiseAction in possible_moves:
                    new_state = state
                    for move in noiseAction:
                        reward, new_state = self.environment.apply_dynamics(new_state, move)
                    if new_state not in self.states:
                        self.states.append(new_state)
                        frontier.append(new_state)

        print("states finished")

        # t model (linear alg)
        t_model = np.zeros([len(self.states), len(constants.ROBOT_ACTIONS), len(self.states)])
        r_model = np.zeros([len(self.states), len(constants.ROBOT_ACTIONS)])
        for state_index, state in enumerate(self.states):
            for action_index, action in enumerate(constants.ROBOT_ACTIONS):
                if self.environment.is_solved(state):
                    t_model[state_index][action_index][state_index] = 1.0
                    r_model[state_index][action_index] = 0.0
                else:
                    expected_value = 0.0
                    for stoch_action, probability in self.stoch_action(action):
                        # Apply action
                        next_state = state
                        reward_array = []
                        for a in stoch_action:
                            reward, next_state = self.environment.apply_dynamics(next_state, a)
                            reward_array.append(reward)
                        expected_value += (probability * min(reward_array))
                        next_state_index = self.states.index(next_state)
                        t_model[state_index][action_index][next_state_index] += probability
                    r_model[state_index][action_index] = expected_value
        self.t_model = t_model
        self.r_model = r_model

        # lin alg policy
        self.la_policy = np.zeros([len(self.states)], dtype=np.int64)
        for i, s in enumerate(self.states):
            self.la_policy[i] = FORWARD
        print("finished initialising")

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R
        state_numbers = np.array(range(len(self.states)))  # indices of every state
        t_pi = self.t_model[state_numbers, self.la_policy]
        r_pi = self.r_model[state_numbers, self.la_policy]
        values = np.linalg.solve(np.identity(len(self.states)) - (self.environment.gamma * t_pi), r_pi)
        self.values = {s: values[i] for i, s in enumerate(self.states)}

        # policy improvement
        new_policy = {s: constants.ROBOT_ACTIONS[self.la_policy[i]] for i, s in enumerate(self.states)}
        for state in self.states:
            # Keep track of maximum value
            action_values = dict()
            for action in constants.ROBOT_ACTIONS:
                total = 0.0
                for possible_action, p in self.stoch_action(action):
                    new_state = state
                    total_reward = 0.0
                    reward_array = []
                    # Calculate V(s) for an action
                    for a in possible_action:
                        reward, new_state = self.environment.apply_dynamics(new_state, a)
                        reward_array.append(reward)
                    total += p * (min(reward_array) + (self.environment.gamma * self.values[new_state]))
                action_values[action] = total

            # Update policy
            new_policy[state] = self.dict_argmax(action_values)

        # check for convergence
        if new_policy == self.policy:
            self.converged = True

        self.policy = new_policy
        for i, s in enumerate(self.states):
            self.la_policy[i] = self.policy[s]

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy.get(state, constants.FORWARD)

    # === Helper Methods ===============================================================================================
    def stoch_action(self, action):
        normal = [action]
        drift_cw = [SPIN_RIGHT, action]
        drift_ccw = [SPIN_LEFT, action]
        double = [action, action]
        drift_cw_double = [SPIN_RIGHT, action, action]
        drift_ccw_double = [SPIN_LEFT, action, action]
        normal_probs = (1 - self.environment.drift_cw_probs[action] - self.environment.drift_ccw_probs[action]) * \
                       (1 - self.environment.double_move_probs[action])
        drift_left_probs = self.environment.drift_ccw_probs[action]
        drift_right_probs = self.environment.drift_cw_probs[action]
        double_probs = self.environment.double_move_probs[action]
        probabilityAction = [(normal, normal_probs),
                             (drift_cw, drift_right_probs * (1 - double_probs)),
                             (drift_ccw, drift_left_probs * (1 - double_probs)),
                             (double, (1 - drift_left_probs - drift_right_probs) * double_probs),
                             (drift_cw_double, drift_right_probs * double_probs),
                             (drift_ccw_double, drift_left_probs * double_probs)]
        return probabilityAction

    def dict_argmax(self, d):
        max_value = max(d.values())
        for k, v in d.items():
            if v == max_value:
                return k
