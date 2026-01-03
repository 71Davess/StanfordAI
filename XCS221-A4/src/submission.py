from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.getDirection() gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    pass
    # ### START CODE HERE ###
    def minimax(state, agentIndex, depth):
      # Terminal test: win/lose or depth reached before Pacman's move
      if state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0):
        return self.evaluationFunction(state)
      # Get number of agents and legal actions and check for no legal actions
      numAgents = state.getNumAgents()
      legalActions = state.getLegalActions(agentIndex)
      if not legalActions:
        return self.evaluationFunction(state)

      # Compute next agent index and depth
      def next_agent_info(agentIndex, depth):
        nextIndex = (agentIndex + 1) % numAgents
        if nextIndex == 0:
          nextDepth = depth + 1
        else:
          nextDepth = depth
        return nextIndex, nextDepth

      if agentIndex == 0:
        # The agent is Pacman if agentIndex == 0, maximizes the score. We start with negative infinity to ensure any value found is higher.
        bestValue = float('-inf') # Initialize to negative infinity
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          # Next agent index and depth define the distance in the tree, which is part of 
          # the input to the minimax function. Vminimax (successor, d = nextIndex, nextDepth) 
          nextIndex, nextDepth = next_agent_info(agentIndex, depth)
          value = minimax(successor, nextIndex, nextDepth)
          bestValue = max(bestValue, value)
        return bestValue
      else:
        # The agents are Ghosts, minimize the score. We start with positive infinity to ensure any value found is lower.
        bestValue = float('inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          nextIndex, nextDepth = next_agent_info(agentIndex, depth)
          value = minimax(successor, nextIndex, nextDepth)
          bestValue = min(bestValue, value)
        return bestValue

    # Choose the action that maximizes the minimax value
    bestScore = float('-inf')
    # Default to STOP so we have a valid move if there are no legal actions (e.g., terminal state).
    bestAction = Directions.STOP
    for action in gameState.getLegalActions(0):
      successor = gameState.generateSuccessor(0, action)
      score = minimax(successor, 1 % gameState.getNumAgents(), 0)
      if score > bestScore:
        bestScore = score
        bestAction = action
    #Print to check the chosen depth and score, and if it matches expected values. Best action added for clarity. 
    print("depth", self.depth, "bestScore", bestScore, "bestAction", bestAction)
    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    pass
    # ### START CODE HERE ###
    def alphabeta(state, agentIndex, depth, alpha, beta):
      if state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0):
        return self.evaluationFunction(state)

      numAgents = state.getNumAgents()
      legalActions = state.getLegalActions(agentIndex)

      if not legalActions:
        return self.evaluationFunction(state)
      #Same next_agent_info function as in minimax
      def next_agent_info(agentIndex, depth):
        nextIndex = (agentIndex + 1) % numAgents
        if nextIndex == 0:
          nextDepth = depth + 1
        else:
          nextDepth = depth 
        return nextIndex, nextDepth

      if agentIndex == 0:
        value = float('-inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          nextIndex, nextDepth = next_agent_info(agentIndex, depth)
          #In alpha-beta pruning, we update the value and check against beta to prune branches.
          #Beta represents the minimum score that the minimizing player is assured.
          #Alpha, on the other side, is the maximum score that the maximizing player is assured.
          #The condition to prune is when the maximizing player's best option (value) exceeds the minimizing player's best option (beta). So if Alpha > = beta, we can prune the remaining branches.
          value = max(value, alphabeta(successor, nextIndex, nextDepth, alpha, beta))
          if value > beta:
            return value
          alpha = max(alpha, value)
        return value
      else:
        value = float('inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          nextIndex, nextDepth = next_agent_info(agentIndex, depth)
          value = min(value, alphabeta(successor, nextIndex, nextDepth, alpha, beta))
          if value < alpha:
            return value
          beta = min(beta, value)
        return value

    bestScore = float('-inf')
    bestAction = Directions.STOP
    alpha = float('-inf')
    beta = float('inf')

    for action in gameState.getLegalActions(0):
      successor = gameState.generateSuccessor(0, action)
      score = alphabeta(successor, 1 % gameState.getNumAgents(), 0, alpha, beta)
      if score > bestScore:
        bestScore = score
        bestAction = action
      alpha = max(alpha, bestScore)

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    pass
    # ### START CODE HERE ###
    def expectimax(state, agentIndex, depth):
      if state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0):
        return self.evaluationFunction(state)

      numAgents = state.getNumAgents()
      legalActions = state.getLegalActions(agentIndex)

      if not legalActions:
        return self.evaluationFunction(state)

      def next_agent_info(agentIndex, depth):
        nextIndex = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextIndex == 0 else depth
        return nextIndex, nextDepth

      if agentIndex == 0:
        value = float('-inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          nextIndex, nextDepth = next_agent_info(agentIndex, depth)
          value = max(value, expectimax(successor, nextIndex, nextDepth))
        return value
      else:
        total = 0.0
        nextIndex, nextDepth = None, None
        probability = 1.0 / len(legalActions)
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          nextIndex, nextDepth = next_agent_info(agentIndex, depth)
          total += probability * expectimax(successor, nextIndex, nextDepth)
        return total

    bestScore = float('-inf')
    bestAction = Directions.STOP

    for action in gameState.getLegalActions(0):
      successor = gameState.generateSuccessor(0, action)
      score = expectimax(successor, 1 % gameState.getNumAgents(), 0)
      if score > bestScore:
        bestScore = score
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """
  pass
  # ### START CODE HERE ###
  if currentGameState.isWin():
    return float('inf')
  if currentGameState.isLose():
    return float('-inf')

  position = currentGameState.getPacmanPosition()
  food = currentGameState.getFood().asList()
  capsules = currentGameState.getCapsules()
  ghostStates = currentGameState.getGhostStates()

  score = currentGameState.getScore()

  # Food features
  if food:
    foodDistances = [manhattanDistance(position, f) for f in food]
    closestFood = min(foodDistances)
    score += 1.5 / (closestFood + 1.0)
    score -= 0.5 * len(food)

  # Capsule incentive
  score -= 2.0 * len(capsules)

  # Ghost features
  for ghost in ghostStates:
    ghostPos = ghost.getPosition()
    distance = manhattanDistance(position, ghostPos)
    if ghost.scaredTimer > 0:
      # Encourage chasing scared ghosts that are near
      score += 2.0 * ghost.scaredTimer / (distance + 1.0)
    else:
      # Penalize proximity to active ghosts
      if distance > 0:
        score -= 4.0 / distance

  return score
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction
