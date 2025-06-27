"""
CS311 Programming Assignment 2: Adversarial Search

Full Name: Sujay Banerjee

Brief description of my evaluation function:

TODO Briefly describe your evaluation function and why it improves the win rate

My evaluation function is better than the  default evaluation function because it considers the following factors:
- Distance to food: Pacman is rewarded for being close to food
- Distance to ghost: Pacman is penalized if too close to active ghost
- Scared ghosts: Pacman is rewarded if close to scared ghost
- Capsules: Pacman is rewarded for being close to capsules
- Number of legal moves: Pacman is rewarded for having more legal moves
- Winning and losing: Pacman is heavily penalized for losing and heavily rewarded for winning

"""

import math, random, typing

import util
from game import Agent, Directions
from pacman import GameState



class ReflexAgent(Agent):
    """
    A reflex agent chooses the best action at each choice point by examining its alternatives via a state evaluation
    function.

    The code below is provided as a guide. You are welcome to change it as long as you don't modify the method
    signatures.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState) -> str:
        """Choose the best action according to an evaluation function.

        Review pacman.py for the available methods on GameState.

        Args:
            gameState (GameState): Current game state

        Returns:
            str: Chosen legal action in this state
        """
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Compute the score for the successor states, choosing the highest scoring action
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Break ties randomly
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, gameState: GameState, action: str):
        """Compute score for current game state and proposed action"""
        successorGameState = gameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()


def scoreEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState (as shown in Pac-Man GUI)

    This is the default evaluation function for adversarial search agents (not reflex agents)
    """
    return gameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    Abstract Base Class for Minimax, AlphaBeta and Expectimax agents.

    You do not need to modify this class, but it can be a helpful place to add attributes or methods that used by
    all your agents. Do not remove any existing functionality.
    """

    def __init__(self, evalFn=scoreEvaluationFunction, depth=2):
        self.index = 0  # Pac-Man is always agent index 0
        self.evaluationFunction = globals()[evalFn] if isinstance(evalFn, str) else evalFn
        self.depth = int(depth)
        self.initial_minimax_printed = False  # Add a flag to control printing of minimax value
    #     self.lastPositions = [] # Keep track of last positions for oscillation penalty

    # def updateLastPositions(self, pacman_position):
    #     self.lastPositions.append(pacman_position)
    #     if len(self.lastPositions) > 3:  # Only keep the last 3 positions
    #         self.lastPositions.pop(0)

class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax Agent"""


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """

        """
        Some potentially useful methods on GameState (recall Pac-Man always has an agent index of 0, the ghosts >= 1):

        getLegalActions(agentIndex): Returns a list of legal actions for an agent
        generateSuccessor(agentIndex, action): Returns the successor game state after an agent takes an action
        getNumAgents(): Return the total number of agents in the game
        getScore(): Return the score corresponding to the current state of the game
        isWin(): Return True if GameState is a winning state
        gameState.isLose(): Return True if GameState is a losing state
        """
        # TODO: Implement your Minimax Agent

        # # keep track of pacman's position
        # pacman_position = gameState.getPacmanPosition()
        # self.updateLastPositions(pacman_position)

        legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_score = float('-inf') #negative infinity bc maximizing

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action) 
            action_score = self.minimax(successor, 0, 1) #agent 1 is ghost
            if action_score > best_score:
                best_score = action_score
                best_action = action

        # Print the Minimax value for the initial state 
        if not self.initial_minimax_printed:
            print(f"Minimax value at depth {self.depth}: {best_score}")
            self.initial_minimax_printed = True  # Set flag to True to prevent further printing

        return best_action
        

    def minimax(self, state: GameState, depth: int, agentIndex: int) -> float:
        # base case
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        # pac-man's turn -- max
        if agentIndex == 0:
            return self.max_value(state, depth)
        # ghosts' turn -- min
        else:
            return self.min_value(state, depth, agentIndex)
        
    
    def max_value(self, state: GameState, depth: int) -> float:
        legal_actions = state.getLegalActions(0)  # Pac-Man is agent 0
        if not legal_actions:
            return self.evaluationFunction(state)

        # Get max value of successors
        max_eval = float('-inf') #negative infinity bc maximizing
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            max_eval = max(max_eval, self.minimax(successor, depth, 1)) #agent 1 is ghost
        return max_eval

    def min_value(self, state: GameState, depth: int, agentIndex: int) -> float:
        legal_actions = state.getLegalActions(agentIndex)
        if not legal_actions:
            return self.evaluationFunction(state)

        min_eval = float('inf') #positive infinity bc minimizing
        num_agents = state.getNumAgents()
        if agentIndex == num_agents - 1: # last ghost
            next_agent = 0  # pac-man moves 
            next_depth = depth + 1 # all agents moved
        else:
            next_agent = agentIndex + 1  # move to next ghost
            next_depth = depth  

        # Get min value of successors for ghost agents
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            min_eval = min(min_eval, self.minimax(successor, next_depth, next_agent))
        return min_eval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action with alpha-beta pruning from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        # TODO: Implement your Minimax Agent with alpha-beta pruning

        # # keep track of pacman's position
        # pacman_position = gameState.getPacmanPosition()
        # self.updateLastPositions(pacman_position)

        alpha = float('-inf')  #pacman
        beta = float('inf')  #ghosts
        legal_actions = gameState.getLegalActions(0)  
        best_action = None
        best_score = float('-inf') 

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action) 
            action_score = self.alpha_beta(successor, 0, 1, alpha, beta)
            if action_score > best_score:
                best_score = action_score
                best_action = action
            alpha = max(alpha, best_score)

        # Print the Minimax value for the initial state 
        if not self.initial_minimax_printed:
            print(f"Minimax value at depth {self.depth}: {best_score}")
            self.initial_minimax_printed = True  # Set flag to True to prevent further printing

        return best_action


    def alpha_beta(self, state: GameState, depth: int, agentIndex: int, alpha: float, beta: float) -> float:
        # base case
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        
        if agentIndex == 0:
            return self.max_value_aB(state, depth, alpha, beta)
        else:
            return self.min_value_aB(state, depth, agentIndex, alpha, beta)
        

    def max_value_aB(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        legal_actions = state.getLegalActions(0)
        if not legal_actions:
            return self.evaluationFunction(state)
        
        max_eval = float('-inf') #negative infinity bc maximizing

        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            max_eval = max(max_eval, self.alpha_beta(successor, depth, 1, alpha, beta)) #agent 1 is ghost
            if max_eval >= beta: # if max_eval is greater than beta, ghosts (min) will never choose this path, so prune
                return max_eval
            alpha = max(alpha, max_eval) # update alpha, best score for pacman so far
        return max_eval
    

    def min_value_aB(self, state: GameState, depth: int, agentIndex: int, alpha: float, beta: float) -> float:
        legal_actions = state.getLegalActions(agentIndex)
        if not legal_actions:
            return self.evaluationFunction(state)

        min_eval = float('inf') #positive infinity bc minimizing

        num_agents = state.getNumAgents()
        if agentIndex == num_agents - 1: # last ghost
            next_agent = 0  # pac-man moves 
            next_depth = depth + 1 # all agents moved
        else:
            next_agent = agentIndex + 1  # move to next ghost
            next_depth = depth  
        
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            min_eval = min(min_eval, self.alpha_beta(successor, next_depth, next_agent, alpha, beta))
            if min_eval <= alpha: # if min_eval is less than alpha, pacman (max) will never choose this path, so prune
                return min_eval
            beta = min(beta, min_eval) # update beta, best score for ghosts so far
        return min_eval
    



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent
    """
    def getAction(self, gameState):
        """Return the expectimax action from the current gameState.

        All ghosts should be modeled as choosing uniformly at random from their legal moves.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        # TODO: Implement your Expectimax Agent

        # # keep track of pacman's position
        # pacman_position = gameState.getPacmanPosition()
        # self.updateLastPositions(pacman_position)

        legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_score = float('-inf') 
        
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            action_score = self.expectimax(successor, 0, 1)
            if action_score > best_score:
                best_score = action_score
                best_action = action
        return best_action


    def expectimax(self, state: GameState, depth: int, agentIndex: int) -> float:
        # base case
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0: #pacman
            return self.max_value(state, depth)
        else:
            return self.expected_value(state, depth, agentIndex) #ghosts
        
    def max_value(self, state: GameState, depth: int) -> float:
        legal_actions = state.getLegalActions(0)
        if not legal_actions:
            return self.evaluationFunction(state)
        
        max_eval = float('-inf') 

        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            max_eval = max(max_eval, self.expectimax(successor, depth, 1))
        return max_eval

    def expected_value(self, state: GameState, depth: int, agentIndex: int) -> float:
        legal_actions = state.getLegalActions(agentIndex)
        if not legal_actions:
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        num_actions = len(legal_actions)
        total = 0 
        if agentIndex == num_agents - 1:
            next_agent = 0
            next_depth = depth + 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            total += self.expectimax(successor, next_depth, next_agent) #sum of all possible outcomes
        return total / num_actions #expected value



def betterEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState using custom evaluation function that improves agent performance.
    """

    """
    The evaluation function takes the current GameStates (pacman.py) and returns a number,
    where higher numbers are better.

    Some methods/functions that may be useful for extracting game state:
    gameState.getPacmanPosition() # Pac-Man position
    gameState.getGhostPositions() # List of ghost positions
    gameState.getFood().asList() # List of positions of current food
    gameState.getCapsules() # List of positions of current capsules
    gameState.getGhostStates() # List of ghost states, including if current scared (via scaredTimer)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    """

    # TODO: Implement your evaluation function
    
    pacman_position = gameState.getPacmanPosition()
    score = gameState.getScore() 
    food = gameState.getFood().asList()
    capsules = gameState.getCapsules()
    ghost_states = gameState.getGhostStates()
    
    # Reward based on distance to food
    if food:
        min_food_distance = min(util.manhattanDistance(pacman_position, food_pos) for food_pos in food)
        score += 100.0 / (min_food_distance + 1) 
        score -= len(food) * 20  # Penalize for remaining food item


    # Eat scared ghosts and avoid active ghosts
    for ghost in ghost_states:
        ghost_position = ghost.getPosition()
        distance_to_ghost = util.manhattanDistance(pacman_position, ghost_position)

        if ghost.scaredTimer > 0:
            # close to scared ghost
            score += 100.0 / (distance_to_ghost + 1)
        else:
            # stay away from active ghost
            if distance_to_ghost < 2:
                score -= 1000  # Heavy penalty, too close to active ghost
            else:
                score -= 150.0 / distance_to_ghost


    # Eat capsules to scare ghosts
    if capsules:
        min_capsule_distance = min(util.manhattanDistance(pacman_position, capsule) for capsule in capsules)
        if any(ghost.scaredTimer == 0 for ghost in ghost_states):  # If any ghost is active
            score += 30.0 / (min_capsule_distance + 1)

   

    # Reward for more legal moves
    legal_moves = gameState.getLegalActions(0)
    score += len(legal_moves) * 10

    # Reward for winning and penalty for losing
    if gameState.isWin():
        score += 1000000
    if gameState.isLose():
        score -= 1000000

    return score



# Create short name for custom evaluation function
better = betterEvaluationFunction
