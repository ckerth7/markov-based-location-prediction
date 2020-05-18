import numpy as np
from numpy.linalg import matrix_power

class MarkovChain:
  def __init__(self, states, initial_cluster, transition_matrix, number_of_occurrences, state_sequence, initial_distribution):
    self.transition_matrix = transition_matrix
    self.states = states
    self.initial_cluster = initial_cluster
    self.number_of_occurrences = number_of_occurrences
    self.state_sequence = state_sequence
    self.initial_distribution = initial_distribution
    self.idv = np.zeros(3)

  # Returns initial cluster as text
  def startState(self):
    if self.initial_cluster == 0:
      return 'd'
    return 'c' + str(self.initial_cluster)

  def determineStart(self):
    return np.argmax(self.initial_distribution)

  def nextStateDpm(self, current_state):
    state_probabilities = self.transition_matrix[int(current_state)]
    return np.argmax(state_probabilities)

  def nextStateIdv(self):
    self.idv = self.idv.dot(self.transition_matrix)
    return np.argmax(self.idv)

  def ResetIDV(self):
      self.idv[0] = 0
      self.idv[1] = 1
      self.idv[2] = 0

  def printTransitionMatrix(self):
    text = "     |  d  |  c  |  r \n"
    text += "  d   |  " + format(self.transition_matrix[0][0], '.2f') + "  "
    text += "|  " + format(self.transition_matrix[0][1], '.2f') + "  "
    text += "|  " + format(self.transition_matrix[0][2], '.2f') + "  " + "\n"
    text += "  c   |  " + format(self.transition_matrix[1][0], '.2f') + "  "
    text += "|  " + format(self.transition_matrix[1][1], '.2f') + "  "
    text += "|  " + format(self.transition_matrix[1][2], '.2f') + "  " + "\n"
    text += "  r   |  " + format(self.transition_matrix[2][0], '.2f') + "  "
    text += "|  " + format(self.transition_matrix[2][1], '.2f') + "  "
    text += "|  " + format(self.transition_matrix[2][2], '.2f') + "  " + "\n"
    print(text)

  def TransitionMatrixToText(self):
    text = "|"
    for i in range(len(self.transition_matrix)):
      for j in range(len(self.transition_matrix)):
        if j < len(self.transition_matrix) - 1:
          text += format(self.transition_matrix[i][j], '.2f') + "|"
        else:
          text += format(self.transition_matrix[i][j], '.2f') + " | "
        # text += str(self.transition_matrix[i][j]) + "|"
    return text

  def InitialStateDistributionToText(self):
    text = "|"
    for i in range(len(self.initial_distribution)):
      if i < len(self.initial_distribution) - 1:
        text += format(self.initial_distribution[i], '.2f') + "|"
      else:
        text += format(self.initial_distribution[i], '.2f') + " | "
      # text += str(self.initial_distribution[i]) + "|"
    return text
    