from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
  # a code that simulates casino machines (bandit)
  # goal: pick 3 bandits and find the best moment to play in each one

  def __init__(self, m):
    self.m = m
    self.mean = 0 # estimate of bandits mean
    self.N = 0    # number of plays

  def pull(self):
    # simulates pulling the bandits arm
    return np.random.randn() + self.m

  def update(self, x):
    # x is the latest sample received from the bandit
    self.N += 1
    self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

def run_experiment(m1, m2, m3, eps, N):
  # there's 3 different means because we compare 3 bandits in this example
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
  data = np.empty(N)
  
  for i in xrange(N):
    # epsilon greedy

    # generate a random number P between 0 and 1
    p = np.random.random()

    if p < eps:
      # choose a bandit at random
      j = np.random.choice(3)
    else:
      # choose the bandit with the best current sample mean
      j = np.argmax([b.mean for b in bandits])

    # pulling the bandit
    x = bandits[j].pull()

    # update the bandit with the reward x got
    bandits[j].update(x)

    # for the plot
    data[i] = x

  # calculate the cumulative average
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average)
  plt.plot(np.ones(N) * m1)
  plt.plot(np.ones(N) * m2)
  plt.plot(np.ones(N) * m3)
  plt.xscale('log')
  plt.show()

  for b in bandits:
    print(b.mean)

  return cumulative_average


if __name__ == '__main__':
  # do the same experiment 3 times, with different espilons

  # when epsilon is 10%
  c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)

  # when epsilon is 5%
  c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)

  # when epsilon is 1%
  c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.xscale('log')
  plt.show()

  # linear plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.show()

