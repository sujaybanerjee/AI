import abc
from collections import defaultdict
import random
from typing import Dict, List
import math

class ReinforcementLearner(metaclass=abc.ABCMeta):
    """Represents an abstract reinforcement learning agent."""

    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, **kwargs):
        """Initialize GridWorld reinforcement learning agent.

        Args:
            numStates (int): Number of states in the MDP.
            numActions (int): Number of actions for each state in the MDP.
            epsilon (float): Probability of taking a random action.
            gamma (float): Discount parameter.
        """
        self.numStates = numStates
        self.numActions = numActions

        self.epsilon = epsilon
        self.gamma = gamma

        
    @abc.abstractmethod
    def action(self, state: int) -> int:
        """Return learned action for the given state."""
        pass

    @abc.abstractmethod
    def epsilonAction(self, step: int, state: int) -> int:
        """With probability epsilon returns a uniform random action. Otherwise return learned action for given state."""
        pass

    def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform the last learning step of an episode. 

        Args:
            step (int): Index of the current step.
            curState (int): Current state, e.g., s
            action (int): Current action, e.g., a
            reward (float): Observed reward
            nextState (int): Next state, e.g., s'. Since this is a terminal step, this is a terminal state.
        """
        self.learningStep(step, curState, action, reward, nextState)

    @abc.abstractmethod
    def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform a learning step of an episode. 

        Args:
            step (int): Index of the current step.
            curState (int): Current state, e.g., s
            action (int): Current action, e.g., a
            reward (float): Observed reward
            nextState (int): Next state, e.g., s'.
        """
        pass


class ModelBasedLearner(ReinforcementLearner):
    """Model-based value iteration reinforcement learning agent."""
    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, updateIter: int = 1000, valueConvergence: float = .001, **kwargs):
        super().__init__(numStates, numActions, epsilon, gamma)
        
        self.updateIter = updateIter
        self.valueConvergence = valueConvergence

        # Maintain transition counts and total rewards for each (s, a, s') triple as a list-of-lists-of dictionaries
        # indexed first by state, then by actions. The keys are in the dictionaries are s'.
        self.tCounts: List[List[defaultdict]] = []
        self.rTotal : List[List[defaultdict]]= []
        for _ in range(numStates):
            self.tCounts.append([defaultdict(int) for _ in range(numActions)])
            self.rTotal.append([defaultdict(float) for _ in range(numActions)])

        # Current policy implemented as a dictionary mapping states to actions. Only states with a current policy
        # are in the dictionary. Other states are assumed to have a random policy.
        self.pi: Dict[int, int] = {}

    def action(self, state: int) -> int:
        """Return the action in the current policy for the given state."""
        # Return the specified action in the current policy if it exists, otherwise return
        # a random action
        return self.pi.get(state, random.randint(0, self.numActions - 1))


    def epsilonAction(self, step: int, state: int) -> int:
        """With some probability return a uniform random action. Otherwise return the action in the current policy for the given state."""
        # Implement epsilon action selection
        # Implement epsilon decay
        # decay_rate = 1.001  # Adjust decay 
        # self.epsilon *= decay_rate
        decay_rate = .00005
        self.epsilon = self.epsilon + math.exp(-decay_rate * step)
        if random.random() < self.epsilon:
            return random.randint(0, self.numActions - 1)  # exploration: take a random action
        else:
            return self.action(state)  # exploitation: take the policy-suggested action


    def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform a value-iteration learning step for the given transition and reward."""
    
        # Update the observed transitions and rewards for (s, a, s') triples. Since we are using
        # defaultdicts we don't need to check if the key exists before incrementing.
        self.tCounts[curState][action][nextState] += 1  #Count of the transition (s,a,s')
        self.rTotal[curState][action][nextState] += reward  #Total reward of the transition (s,a,s')

        # Update the current policy every updateIter steps
        if step % self.updateIter != 0:
            return
       
        # Implement value iteration to update the policy. 
        # Recall that:
        #   T(s, a, s') = (Counts of the transition (s,a) -> s') / (total transitions from (s,a))
        #   R(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
        # Many states may not have been visited yet, so we need to check if the counts are zero before
        # updating the policy. We will only update the policy for states with state-action pairs that\
        # have been visited.
   
        # Recall value iteration is an iterative algorithm. Here iterate until convergence, i.e., when
        # the change between v_new and v is less than self.valueConvergence for all states.
        v = [0.0] * self.numStates
        while True:
            v_new = v[:] # Make a copy of current values to update

            # Calculate v_new for each state for which you have observed transitions
            for state in range(self.numStates):
                max_value = float('-inf')  # Initialize max value for this state
                has_observed_action = False  # Flag to check if any action has transitions

                for action in range(self.numActions):
                    # Calculate T(s, a, s') and R(s, a, s')
                    total_transitions = sum(self.tCounts[state][action].values())
                    if total_transitions == 0:
                        continue  # Skip if no transitions observed

                    has_observed_action = True
                    expected_value = 0.0  # Initialize expected value for action
                    for next_state, count in self.tCounts[state][action].items():
                        probability = count / total_transitions  # estimate transition probability
                        reward = self.rTotal[state][action][next_state] / count  # estimate reward
                        expected_value += probability * (reward + self.gamma * v[next_state])

                    max_value = max(max_value, expected_value)  # choose action with the highest value
                if has_observed_action:
                    v_new[state] = max_value  # Update value of the state
                else:
                    v_new[state] = v[state]  # Keep value unchanged if no actions observed

            # Change in values?       
            if all(abs(new - prev) <= self.valueConvergence for new, prev in zip(v_new, v)):
                break
            v = v_new             
       
        # Update policy based on results of value iteration      
        for state in range(self.numStates):
            max_action_value = float('-inf')
            best_action = None
            for action in range(self.numActions):
                total_transitions = sum(self.tCounts[state][action].values())
                if total_transitions == 0:
                    continue  # skip if no transitions observed

                action_value = 0.0
                for next_state, count in self.tCounts[state][action].items():
                    probability = count / total_transitions
                    reward = self.rTotal[state][action][next_state] / count
                    action_value += probability * (reward + self.gamma * v[next_state])

                if action_value > max_action_value:
                    max_action_value = action_value
                    best_action = action

            if best_action is not None:
                self.pi[state] = best_action # Store best action for the state in the policy

class QLearner(ReinforcementLearner):
    """Q-learning-based reinforcement learning agent."""
    
    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, alpha: float = 0.1, initQ: float=0.0, **kwargs):
        """Initialize GridWorld reinforcement learning agent.

        Args:
            numStates (int): Number of states in the MDP.
            numActions (int): Number of actions for each state in the MDP.
            epsilon (float): Probability of taking a random action.
            gamma (float): Discount parameter.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            initQ (float, optional): Initial Q value. Defaults to 0.0.
        """
        super().__init__(numStates, numActions, epsilon=epsilon, gamma=gamma)

        self.alpha = alpha

        # The Q-table, q, is a list-of-lists, indexed first by state, then by actions
        self.q: List[List[float]] = []  
        for _ in range(numStates):
            self.q.append([initQ] * numActions)

    def action(self, state: int) -> int:
        """Returns a greedy action with respect to the current Q function (breaking ties randomly)."""
        # Implement greedy action selection
        max_q_value = max(self.q[state])  # max Q-value for the state
        best_actions = [action for action, q_value in enumerate(self.q[state]) if q_value == max_q_value] # Find actions with max Q-value to break ties
        return random.choice(best_actions)  # break ties randomly

    def epsilonAction(self, step: int, state: int) -> int:
        """With probability epsilon returns a uniform random action. Otherwise it returns a greedy action with respect to the current Q function (breaking ties randomly)."""
        # Implement epsilon-greedy action selection
        if random.random() < self.epsilon:  # random action to explore
            return random.randint(0, self.numActions - 1)
        else:  # best action according to current Q-values to exploit
            return self.action(state)

    def learningStep(self, step: int, curState, action, reward, nextState):
        """Performs a Q-learning step based on the given transition, action and reward."""
        #  Implement the Q-learning step
        current_q = self.q[curState][action]  # current Q-value for (curState, action)
        max_next_q = max(self.q[nextState])  # max Q-value for nextState
        # Compute TD and update the Q-value for (curState, action)
        self.q[curState][action] += self.alpha * (reward + self.gamma * max_next_q - current_q)

    def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Performs the last learning step of an episode. Because the episode has terminated, the next Q-value is 0."""
        # Implement the terminal step of the learning algorithm
        current_q = self.q[curState][action]  # current Q-value for (curState, action)
        self.q[curState][action] += self.alpha * (reward - current_q)  # next Q-value is 0
