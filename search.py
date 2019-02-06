# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class SearchNode():
    def __init__(self, node, parent=None):
        self.state = node[0]
        self.direction = node[1]
        self.parent = parent
        

def genericSearch(problem, fringe):
    # every element in fringe contains all past positions and past directions.
    """
    fringe.push([[problem.getStartState()], []])
    while not fringe.isEmpty():
        node = fringe.pop()
        currentLoc = node[0][-1]
        prevLocs, prevDirs = node[0], node[1]
        # if expanded the goal, return list of direction.
        if problem.isGoalState(currentLoc):
            return prevDirs
        # for every successor, check if visited node before.
        # if not, push to fringe with current position and directions.
        for successor in problem.getSuccessors(currentLoc):
            nextLoc, nextDir = successor[0], successor[1]
            if nextLoc not in prevLocs:
                updatedLocs, updatedDirs = prevLocs + [nextLoc], prevDirs + [nextDir]
                fringe.push([updatedLocs, updatedDirs])
    return []
    """
    """
    start = problem.getStartState()
    prevLocs = []   # track the previously expanded nodes
    locs_map, dirs_map = dict(), dict() # map child-parent and node-direction
    node_map = []
    fringe.push(start)
    while not fringe.isEmpty():
        node = fringe.pop()
        
        # if expanded the goal, return list of direction.
        if problem.isGoalState(node):
            dirs, ptr = [], node
            #print(locs_map)
            while ptr != start:
                drtn = dirs_map[ptr]   # get direction from parent to child
                dirs.append(drtn)
                parent = locs_map[ptr] # get parent of current node
                ptr = parent
            dirs.reverse()
            return dirs
        
        # for every successor, check if visited node before.
        # if not, push to fringe with current position and directions.
        if node not in prevLocs:
            prevLocs.append(node)
            for successor in problem.getSuccessors(node):
                successorLoc, dirToSuccessor = successor[0], successor[1]
                #if successorLoc not in prevLocs or successorLoc not in fringe.list:
                if successorLoc not in prevLocs:
                    #node_map.append(SearchNode(successor, parent=node))
                    locs_map[successorLoc] = node  # buggy here...works for trees but not general graphs where a child can have multiple parents
                    dirs_map[successorLoc] = dirToSuccessor # also buggy here for nodes with multiple arrows pointing to it
                    fringe.push(successorLoc)    # only visit node if it hasn't been visited before
                    print("{0} adds {1} to fringe".format(node, successorLoc))
    return []
    """
    
    start = problem.getStartState()
    prevLocs = []   # track the previously expanded nodes
    node_map = dict()
    fringe.push(start)
    while not fringe.isEmpty():
        state = fringe.pop()
        if state is start:
            loc, dirtn = state, None
        else:
            loc, dirtn = state[0], state[1]

        # if expanded the goal, return list of directions
        if problem.isGoalState(loc):
            dirs, ptr = [], state
            while ptr is not start:
                dirs.append(ptr[1])
                ptr = node_map[ptr]
            dirs.reverse()
            return dirs
        
        # push successors to fringe
        if loc not in prevLocs:
            prevLocs.append(loc)
            for successor in problem.getSuccessors(loc):
                successorLoc, dirToSuccessor = successor[0], successor[1]
                node_map[successor] = state
                fringe.push(successor)    # only visit node if it hasn't been visited before
    return []
    

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Stack
    fringe = Stack()
    return genericSearch(problem, fringe)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    fringe = Queue()
    return genericSearch(problem, fringe)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    fringe = PriorityQueue()
    start = problem.getStartState()
    prevLocs = []   # track the previously expanded nodes
    node_map = dict()
    fringe.push(start, 0)
    while not fringe.isEmpty():
        state = fringe.pop()
        if state is start:
            loc, dirtn, cost = state, None, 0
        else:
            loc, dirtn, cost = state[0], state[1], state[2]

        # if expanded the goal, return list of directions
        if problem.isGoalState(loc):
            dirs, ptr = [], state
            while ptr is not start:
                dirs.append(ptr[1])
                ptr = node_map[ptr]
            dirs.reverse()
            return dirs
        
        # push successors to fringe
        if loc not in prevLocs:
            prevLocs.append(loc)
            for successor in problem.getSuccessors(loc):
                successorLoc, dirToSuccessor, costToSuccessor = successor[0], successor[1], successor[2]
                successor[2] += cost
                node_map[successor] = state
                fringe.push(successor, cost + costToSuccessor)    # only visit node if it hasn't been visited before
                print("{0} adds {1} to fringe, total={2}".format(state, successor, cost + costToSuccessor))
    return []
    return genericSearch(problem, fringe)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueueWithFunction
    g, h = problem.costFn, heuristic
    f = lambda n: g(n) + h(n, problem)
    fringe = PriorityQueueWithFunction(f)
    return genericSearch(problem, fringe)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
