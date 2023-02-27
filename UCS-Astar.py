import constants
from environment import *
import heapq


class Node:
    def __init__(self, state, cost, actions):
        self.state = state
        self.cost = cost
        self.actions = actions

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        else:
            return self.actions[len(self.actions) - 1] < other.actions[len(other.actions) - 1]


class Solver:

    def __init__(self, environment, loop_counter):
        self.environment = environment
        self.loop_counter = loop_counter

    def solve_ucs(self):
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        :return: path (list of actions, where each action is an element of ROBOT_ACTIONS)
        """
        frontier = [Node(self.environment.get_init_state(), 0, [])]
        visited = {self.environment.get_init_state(): 0}  # {state: cost}
        heapq.heapify(frontier)  # initialise priority queue

        while len(frontier) > 0:  # while queue isn't empty
            self.loop_counter.inc()
            node = heapq.heappop(frontier)

            # check if goal state
            if self.environment.is_solved(node.state):
                print(f"UCS Nodes before termination {len(frontier)}")
                print(f"UCS Nodes visited before termination {len(visited)}")
                return node.actions

            successors = self.get_successors(node.state)
            for successor in successors:  # state = (nextState, cost)
                newCost = node.cost + successor["cost"]
                if (successor["state"] not in visited.keys()) or (newCost < visited[successor["state"]]):
                    visited[successor["state"]] = newCost
                    heapq.heappush(frontier, Node(successor["state"], newCost, node.actions + [successor["action"]]))

    def solve_a_star(self):
        """
        Find a path which solves the environment using A* search.
        :return: path (list of actions, where each action is an element of ROBOT_ACTIONS)
        """
        frontier = [(self.heuristics(self.environment, self.environment.get_init_state()), Node(self.environment.get_init_state(), 0, []))]
        visited = {self.environment.get_init_state(): 0}  # {state: cost}
        heapq.heapify(frontier)  # initialise priority queue

        while len(frontier) > 0:  # while queue isn't empty
            self.loop_counter.inc()
            _, node = heapq.heappop(frontier)

            # check if goal state
            if self.environment.is_solved(node.state):
                print(f"A* Nodes before termination {len(frontier)}")
                print(f"A* Nodes visited before termination {len(visited)}")
                return node.actions

            successors = self.get_successors(node.state)
            for successor in successors:  # state = (nextState, cost)
                newCost = node.cost + successor["cost"]
                if (successor["state"] not in visited.keys()) or (newCost < visited[successor["state"]]):
                    visited[successor["state"]] = newCost
                    heapq.heappush(frontier, (newCost + self.heuristics(self.environment, node.state), Node(successor["state"], newCost, node.actions + [successor["action"]])))


    def heuristics(self, environment, state):
        filledTarget = 0
        totalTargets = len(environment.target_list)
        widget_cells = [widget_get_occupied_cells(environment.widget_types[i], state.widget_centres[i],
                                                  state.widget_orients[i]) for i in range(environment.n_widgets)]

        # for each widget
        # find the closest target to that widget
        # get the distance, append to list

        distanceToTarget = []
        # find closest target for each widget
        for widget in state.widget_centres:
            closestTarget = []
            for tgt in environment.target_list:
                targetRow = tgt[0]
                targetCol = tgt[1]
                targetCoord = (targetRow, targetCol)
                closestTarget.append(self.hexagonal_distance(widget, targetCoord))

            distanceToTarget.append(min(closestTarget))

        # loop over each target
        for tgt in environment.target_list:
            # loop over all widgets to find a match
            for i in range(environment.n_widgets):
                if tgt in widget_cells[i]:
                    # match found
                    filledTarget += 1
                    break

        targetsUnoccupied = totalTargets - filledTarget
        hexDistance = sum(distanceToTarget)
        return hexDistance + targetsUnoccupied

    # Distance formula from https://stackoverflow.com/a/15926930
    def hexagonal_distance(self, p1, p2):
        y1, x1 = p1
        y2, x2 = p2
        du = x2 - x1
        dv = (y2 + x2 // 2) - (y1 + x1 // 2)

        if (du >= 0 and dv >= 0) or (du < 0 and dv < 0):
            return max(abs(du), abs(dv))
        else:
            return abs(du) + abs(dv)

    def get_successors(self, state):
        successors = []
        for action in constants.ROBOT_ACTIONS:
            successful, cost, nextState = self.environment.perform_action(state, action)
            if successful:
                successors.append({"state": nextState, "cost": cost, "action": action})
        return successors

