# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
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
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"

  frontier = []
  explored_states = set([])
  node = (problem.getStartState(), [], 1)

  if problem.isGoalState(node[0]):
      return node[1]
  frontier.insert(0, node)
  while True:
      if not frontier:
          return None
      node = frontier.pop(0)       
      explored_states.add(node[0])
      successor_candidates = problem.getSuccessors(node[0])
      for candidate in successor_candidates:
          path_to_node = list(node[1])
	  path_to_node.append(candidate[1])
	  child_node = (candidate[0], path_to_node, candidate[2])
          if candidate not in frontier and candidate[0] not in explored_states:
	      if problem.isGoalState(child_node[0]):
	          return child_node[1]
	      frontier.insert(0, child_node)
      
def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  "*** YOUR CODE HERE ***"
 
  frontier = []
  explored_states = set([])
  node = (problem.getStartState(), [], 1)

  if problem.isGoalState(node[0]):
      return node[1]
  frontier.insert(0, node)
  while True:
      if not frontier:
          return None
      node = frontier.pop(0)
      explored_states.add(node[0])
      successor_candidates = problem.getSuccessors(node[0])
      for candidate in successor_candidates:
          path_to_node = list(node[1])
	  path_to_node.append(candidate[1])
	  child_node = (candidate[0], path_to_node, candidate[2])
	  st = lambda x: x[0]
	  frontier_nodes = set([])
	  for nd in frontier:
              frontier_nodes.add(st(nd))
          if (candidate[0] not in frontier_nodes) and (candidate[0] not in explored_states):
	      if problem.isGoalState(child_node[0]):
	          return child_node[1]
	      frontier.append(child_node)
     
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  
  from util import PriorityQueue

  frontier = PriorityQueue()
  explored_states = set([])
  node = (problem.getStartState(), [], 0)

  if problem.isGoalState(node[0]):
      return node[1]
  PriorityQueue.push(frontier, node, node[2])
  while True:
      if not frontier:
          return None
      node = PriorityQueue.pop(frontier)
      explored_states.add(node[0])
      successor_candidates = problem.getSuccessors(node[0])
      successor_candidates.reverse()
      for candidate in successor_candidates:
	  if candidate[0] not in explored_states:
	      path_to_node = list(node[1])
	      path_to_node.append(candidate[1])
	      child_node = (candidate[0], path_to_node, node[2] + candidate[2])
	      # if candidate[0] not in explored_states:
	      if problem.isGoalState(child_node[0]):
	          return child_node[1]
	      PriorityQueue.push(frontier, child_node, child_node[2])

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"

  from util import PriorityQueue 
 
  frontier = PriorityQueue()
  node = (problem.getStartState(), [], 0)
  PriorityQueue.push(frontier, node, node[2] + heuristic(node[0], problem))
  explored_states = set([])

  if problem.isGoalState(node[0]):
      return node[1]
  elif PriorityQueue.isEmpty(frontier):
      return None
  else:
      while not PriorityQueue.isEmpty(frontier):
          new_node = PriorityQueue.pop(frontier)
          explored_states.add(new_node[0])
	  successor_candidates = problem.getSuccessors(new_node[0])
          if node[0][0] == (13, 5):
	      print "successors: ", successor_candidates
          if successor_candidates:
	      successor_candidates.reverse()
              for candidate in successor_candidates:
		  if candidate[0] not in explored_states:
		      c_path = list(new_node[1])
	              c_path.append(candidate[1])
		      # if candidate[0] not in explored_states:
	              if problem.isGoalState(candidate[0]):
		          return c_path
	              PriorityQueue.push(frontier, (candidate[0], c_path, new_node[2] + candidate[2]), new_node[2] + heuristic(candidate[0], problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
