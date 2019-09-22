# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, operator

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # best scenario: food and no ghost
        # worst scenario: no food and ghost
        # score penalty for naive refelx is the waiting time
        total_evaluation = successorGameState.getScore()
        
        # check if food is in new position 
        total_evaluation += sum([1 / manhattanDistance(newPos, food) for food in newFood.asList()]) 

        # check if ghost is in new position
        total_evaluation += sum([manhattanDistance(newPos, successorGameState.getGhostPosition(i)) for i in range(1, len(newGhostStates))])
        total_evaluation += newScaredTimes[0]
        return total_evaluation

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """   
        action_value_map = dict()
        def value(state, depth, agentIndex):
            nonlocal action_value_map
            # check for depth limit or terminal state
            isTerminalState = state.isWin() or state.isLose()
            if depth == self.depth or isTerminalState:
                return self.evaluationFunction(state)

            # handle min or max node
            agentIndex %= gameState.getNumAgents()
            if agentIndex == 0:
                v = float('-inf')
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    val = value(successor, depth, 1)
                    
                    # store the actions leading to the root's successors
                    if depth == 0:
                        action_value_map[val] = action
                    v = max(v, val)
                return v 
            else:
                v = float('inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)

                    # increment depth if last minimizing agent
                    if agentIndex == gameState.getNumAgents() - 1:
                        v = min(v, value(successor, depth+1, agentIndex+1))
                    else:
                        v = min(v, value(successor, depth, agentIndex+1))
                return v

        # pick the action corresponding to successor w/ largest minimax
        minimax = value(gameState, depth=0, agentIndex=0)
        return action_value_map[minimax]
        """
        minimax_actions = []
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            minimax_actions.append([value(successor, 1, 1), action])    
        return max(minimax_actions, key=lambda x: x[0])[1]
        """
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action_value_map = dict()
        def value(state, depth, agentIndex, a, b):
            nonlocal action_value_map

            # check for depth limit or terminal state
            isTerminalState = state.isWin() or state.isLose()
            if depth == self.depth or isTerminalState:
                return self.evaluationFunction(state)

            # handle min or max node
            agentIndex %= gameState.getNumAgents()
            if agentIndex == 0: # max function
                alpha, beta = a, b
                v = float('-inf') 
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    val = value(successor, depth, 1, alpha, beta)
                    
                    # store the actions leading to the root's successors
                    if depth == 0:
                        action_value_map[val] = action
                    
                    # pruning
                    v = max(v, val)
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v 
            else: # min function
                alpha, beta = a, b
                v = float('inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)

                    # increment depth if last minimizing agent
                    if agentIndex == gameState.getNumAgents() - 1:
                        val = value(successor, depth+1, agentIndex+1, alpha, beta)
                    else:
                        val = value(successor, depth, agentIndex+1, alpha, beta)
                    
                    # pruning
                    v = min(v, val)
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        # pick the action corresponding to successor w/ largest minimax
        minimax = value(gameState, depth=0, agentIndex=0, a=float('-inf'), b=float('inf'))
        return action_value_map[minimax]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action_value_map = dict()
        def value(state, depth, agentIndex):
            nonlocal action_value_map
            # check for depth limit or terminal state
            isTerminalState = state.isWin() or state.isLose()
            if depth == self.depth or isTerminalState:
                return self.evaluationFunction(state)

            # handle max node for pacman and expected utility for other agents
            agentIndex %= gameState.getNumAgents()
            if agentIndex == 0:
                v = float('-inf')
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    val = value(successor, depth, 1)
                    
                    # store the actions leading to the root's successors
                    if depth == 0:
                        action_value_map[val] = action
                    v = max(v, val)
                return v 
            else: # do expectimax with uniform probability
                v = 0
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    p = 1 / (len(state.getLegalActions(agentIndex))) # uniform probability
                    # increment depth if last minimizing agent
                    if agentIndex == gameState.getNumAgents() - 1:
                        v += p * value(successor, depth + 1, agentIndex + 1)
                    else:
                        v += p * value(successor, depth, agentIndex + 1)
                return v

        # pick the action corresponding to successor w/ largest minimax
        expectimax = value(gameState, depth=0, agentIndex=0)
        return action_value_map[expectimax]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    We want our evaluation function to take few things under consideration:
    1. The score in the current position (the higher the better)
    2. the amount of food pellets
    3. The distance from the closest food pellet
    4. ghosts (greater penalty if they are not scared and close, bonus if the are scared)

    -- We don't care about capsuls since they are not essential for winning.

    Basically, we will try to reach our closest food pellet that does not endager us
    using a BFS; the closer we are to a food pellet, the greater value
    a move towards it will achieve. We get bonus for scared ghosts, and penalty
    for not-scared ghosts, since scared ghosts give a score boost that will enhance
    our utility.
    """
    if currentGameState.isLose():
        return float('-inf')
    elif currentGameState.isWin():
        return float('inf')

    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    walls = currentGameState.getWalls().asList()
    score = currentGameState.getScore()
    ghosts = currentGameState.getGhostStates()

    fringe = util.Queue()
    fringe.push(pacmanPos)

    val = score
    visited = [pacmanPos]
    while not fringe.isEmpty(): 
        step = fringe.pop()
        # if we found food, we will not return to the loop
        if step in food:
            for ghost in ghosts:
                # penalise a move near a ghost, give bonus for getting
                # a scared ghost.
                distance = (1/manhattanDistance(ghost.getPosition(), pacmanPos))
                if not ghost.scaredTimer:
                    val -= 90 * distance
                else:
                    val += 25 * distance
            # encourage eating by penalising for more food left.
            val -= 35 * len(food)
            return val
        else:
            # reduce value by one step farther, try to move in every direction that
            # doesn't have a wall.
            val -= 1
            for newPos in [(step[0], step[1] + 1), (step[0] + 1, step[1]),
                                    (step[0] - 1, step[1]), (step[0], step[1] - 1)]:
                if newPos not in visited and newPos not in walls:
                    visited.append(newPos)
                    fringe.push(newPos)
    return float('-inf')

# Abbreviation
better = betterEvaluationFunction
