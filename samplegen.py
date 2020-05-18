import numpy as np
from markov import MarkovChain
import random

class SampleGen:
  def __init__(self, markov_model, rand_number_gen):
    self.transition_matrix = markov_model.transition_matrix
    self.initial_distribution = markov_model.initial_distribution
    self.random = rand_number_gen

  # Returns a sample sequence of state transitions as an array of state indices 
  # and length equal to number_of_steps.
  def sample(self, number_of_steps):
    sample = np.zeros(number_of_steps)
    random_value = self.random.random()
    cum_function = np.cumsum(self.initial_distribution)
    sample[0] = self.index(random_value, cum_function)
    for i in range(1, number_of_steps):
      current_state = sample[i - 1]
      random_value = self.random.random()
      cum_function = np.cumsum(self.transition_matrix[int(current_state)])
      sample[i] = self.index(random_value, cum_function)
    return sample

  # Converts an array of state indices to a list of state names.
  def sample_to_text(self, sample):
    state_sequence = []
    for state_index in sample:
      if (int(state_index) == 0):
        state_sequence.append('d')
      elif (int(state_index) == 1):
        state_sequence.append('c')
      else:
        state_sequence.append('r')
    return state_sequence

  # Converts a list of states to an array of state indices.
  def text_to_sample(self, states):
    state_sequence = np.zeros(len(states))
    for state in states:
      if (state == 'd'):
        state_sequence.append(0)
      elif (state == 'c'):
        state_sequence.append(1)
      else:
        state_sequence.append(2)
    return state_sequence

  # The given cdf is used to create intervals of the form [0, cdf(0)[, [cdf(0), cdf(1)[ ...
  # Given these intervals, this function determines the interval where the uniformly 
  # distributed random value is "falling" into and determines the index of the 
  # state belonging to that interval or -1 in case no matching interval was found.
  # parameters:
  # - random_value: real number from [0, 1[
  # - cdf: cumulative distribution function determined from discrete probability function
  def index(self, random_value, cdf):
    range_index = -1
    for r in range(len(cdf)):
      lower_bound = 0
      if (r > 0):
        lower_bound = cdf[r - 1]
      upper_bound = cdf[r]
      if (random_value >= lower_bound and random_value < upper_bound):
        range_index = r
        break
    return range_index

  def printTransitionMatrix(self):
    text = "     "
    for r in range(len(self.transition_matrix[0])):
      if r == 0:
        text += 'd000'
      elif r < 10:
        text += 'c00' + str(r)
      elif r < 100:
        text += 'c0' + str(r)
      else:
        text += 'c' + str(r)
      text += ' | '
    text += '\n'
    for r in range(len(self.transition_matrix[0])):
      if r == 0:
        text += 'd000'
      elif r < 10:
        text += 'c00' + str(r)
      elif r < 100:
        text += 'c0' + str(r)
      else:
        text += 'c' + str(r)
      for c in range(len(self.transition_matrix[r])):
        text += ' | ' + str(round(self.transition_matrix[r][c], 2))
      text += '| \n'
    print(text)
    