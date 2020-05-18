import numpy as np
from markov import MarkovChain
from samplegen import SampleGen

class Simulation:
    def __init__(self, sample_gen):
        """ Generates a Simulation given a sample generator.
    
        Parameters:
            sample_gen (SampleGen): Used to generate samples
        """
        self.sample_gen = sample_gen

    def runSimulation(self, mc, samples, length, is_dpm):
        """ Runs a simulation based upon the given parameters.
        
        Parameters:
            mc (MarkovChain): The MC that has to be evaluated
            samples (int): Number of generated samples
            length (int): Length of generated samples
            is_dpm (bool): Uses DPM if true or DPM otherwise

        Returns:
            float: Average number of states that were predicted correctly
        """
        ratios = np.zeros(samples)
        for i in range(samples):
            # resets initial distribution in case IDV was used before
            mc.ResetIDV()
            ratios[i] = self.predictSample(mc, length, is_dpm)
        return np.sum(ratios) / samples

    def predictSample(self, mc, length, is_dpm):
        """ Applies a model's prediction onto a generated sample.

        Parameters:
            mc (MarkovChain): The MC that has to be evaluated
            length (int): Length of the generated sample
            is_dpm (bool): Uses DPM if true or DPM otherwise

        Returns:
            float: Ratio of correctly predicted state transitions
        """
        sample = self.sample_gen.sample(length)
        hits = 0
        transitions = len(sample) - 1
        for index in range(0, transitions):
            next_state = sample[index + 1]
            if is_dpm == True:
                predicted_state = mc.nextStateDpm(sample[index])
            else:
                predicted_state = mc.nextStateIdv()
            if (next_state == predicted_state):
                hits += 1
        return hits / transitions

