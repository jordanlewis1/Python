# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'ReflexCaptureAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.myAgents = CaptureAgent.getTeam(self, gameState)
    self.opAgents = CaptureAgent.getOpponents(self, gameState)
    self.myFoods = CaptureAgent.getFood(self, gameState).asList()
    self.agentIndices = sorted(self.myAgents + self.opAgents)
    self.opFoods = CaptureAgent.getFoodYouAreDefending(self, gameState).asList()
    self.treeDepth = 1
    
    '''
    Your initialization code goes here, if you need any.
    '''
  def getSuccessor(self, gameState, action):
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):
         return successor.generateSuccessor(self.index, action)
      else:
         return successor

  def getFeatures(self, gameState, action):
      feats = util.Counter()
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()
      feats['onDefense'] = 1
      if myState.isPacman: feats['onDefense'] = 0
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      feats['numInvaders'] = len(invaders)
      if len(invaders) > 0:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
          feats['invaderDistance'] = min(dists)
          
      if action == Directions.STOP: feats['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: feats['reverse'] = 1
      
      return feats
  
  def getWeights(self, gameState, action):
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
 
  def evaluate(self, gameState, action):
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      return features * weights
  
  def chooseAction(self, gameState):
      agentPos = gameState.getAgentPosition(self.index)
      actions = gameState.getLegalActions(self.index)
      distToFood = []
      for food in self.myFoods:
          distToFood.append(self.distancer.getDistance(agentPos, food))
          
      distToOps = []
      for opponent in self.opAgents:
          opPos = gameState.getAgentPosition(opponent)
          if opPos != None:
              distToOps.append(self.distancer.getDistance(agentPos, opPos))
      
      values = [self.evaluate(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      return random.choice(bestActions)
  
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.foodC = 0
    self.foodA = len(self.getFood(gameState).asList())

      
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    

    foodLeft = len(self.getFood(gameState).asList())

    if self.foodA != foodLeft:
      self.foodC += 1
      self.foodA -= 1
     
    #print(self.foodC)    
      
    features = util.Counter()
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    myState = gameState.getAgentState(self.index)
    if myState.isPacman: features['onDefense'] = 0  
     
    
    if self.foodC >=2:   
      #print("HITTT")
      if features['onDefense'] == 1:
          self.foodC = 0
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
  
    """for action in bestActions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            ops = self.getOpponents(successor)
            closestGhost = 999999, None
            for i in ops:
                ghostPos = successor.getAgentPosition(i)
                ghostDistance = self.getMazeDistance(pos2, ghostPos)
                if ghostDistance < closestGhost[0]:
                    closestGhost = ghostDistance, action"""
    if features['onDefense'] == 0:
        for action in bestActions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            ops = self.getOpponents(successor)
            closestGhost = 999999, None
            for i in ops:
                ghostPos = successor.getAgentPosition(i)
                ghostDistance = self.getMazeDistance(pos2, ghostPos)
                if ghostDistance < closestGhost[0]:
                    closestGhost = ghostDistance, action
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    cPos = successor.getAgentState(self.index).getPosition()
    food = self.getFood(successor).asList()
    features['successorScore'] = -len(food)
    
    enemyPos = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    
    
    enemies = []
    allies = []
    for enemy in enemyPos:
        if enemy.isPacman and enemy.getPosition() is not None:
            enemies.append(enemy)
        elif not enemy.isPacman and enemy.getPosition() is not None:
            allies.append(enemy)
    
    if len(food) > 0 and self.foodC == 0 and len(enemies) == 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, f) for f in food])
      features['distanceToFood'] = minDistance
      
    if self.foodC > 0:
        minDistToH = min([self.getMazeDistance(cPos, self.start)])
        
        features['distHome'] = minDistToH
        
    if action == Directions.STOP:
        features['STOP'] = 1
    
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 100, 'distanceToFood': -1, 'distHome': -10, 'STOP': -100}
    

