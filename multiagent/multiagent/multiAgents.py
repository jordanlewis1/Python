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
import random, util, sys

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
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 1
        closestGhostPos = newGhostStates[0].configuration.pos
        closestGhost = manhattanDistance(newPos, closestGhostPos)
        newFoodPos = newFood.asList()
        #gPosition = childGameState.()
        fDistance = [manhattanDistance(newPos, foodPos) for foodPos in newFoodPos]
        if len(fDistance) == 0:
            return 0
        
        #set the closest food using fdistance
        closestFood = min(fDistance)
        
        if action == "Stop":
            score -= 50
            
            
        return childGameState.getScore() + (closestGhost / (closestFood * 10) + score)
        

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        score = gameState.getNumAgents()
        ActionScore = []
    
        def Stop(List):
          return [x for x in List if x != 'Stop']

        def miniMax(s, iterCount):
          if iterCount >= self.depth*score or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
          if iterCount%score != 0: 
            result = 1e10
            for a in Stop(s.getLegalActions(iterCount%score)):
              sdot = s.getNextState(iterCount%score, a)
              result = min(result, miniMax(sdot, iterCount+1))
            return result
          else: 
            result = -1e10
            for a in Stop(s.getLegalActions(iterCount%score)):
              sdot = s.getNextState(iterCount%score, a)
              result = max(result, miniMax(sdot, iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
          
        result = miniMax(gameState, 0);
        return Stop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]
        
        
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
       



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(gameState)
            if agent == 0: 
                return max(expectiMax(1, depth, gameState.getNextState(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  
                nAgent = agent + 1  
                if gameState.getNumAgents() == nAgent:
                    nAgent = 0
                if nAgent == 0:
                    depth += 1
                return sum(expectiMax(nAgent, depth, gameState.getNextState(agent, newState)) for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            util = expectiMax(1, 0, gameState.getNextState(0, agentState))
            if util > maximum or maximum == float("-inf"):
                maximum = util
                action = agentState

        return action
        
                        
               
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    #childGameState = currentGameState.getPacmanNextState(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    
    "*** YOUR CODE HERE ***"
    

    newFoodList = newFood.asList()
    minFoodD = -1
    for food in newFoodList:
        dist = util.manhattanDistance(newPos, food)
        if minFoodD >= dist or minFoodD == -1:
            minFoodD = dist

    ghostD = 1
    ghostP = 0
    for ghost_state in currentGameState.getGhostPositions():
        dist = util.manhattanDistance(newPos, ghost_state)
        ghostD += dist
        if dist <= 1:
            ghostP += 1

    nCapsule = currentGameState.getCapsules()
    numCapsules = len(nCapsule)

    return currentGameState.getScore() + (1 / float(minFoodD)) - (1 / float(ghostD)) - ghostP - numCapsules
    
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
