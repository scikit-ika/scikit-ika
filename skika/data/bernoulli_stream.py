# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:32:47 2020

@author: tlac980
"""

import numpy as np
from skmultiflow.utils import check_random_state
#import itertools
import math


#from skmultiflow.core import BaseSKMObject
#from abc import ABCMeta, abstractmethod

class BernoulliStream():
    """ A class to generate a Bernoulli Stream
         
         The stream is simulating the error rate of a learner.
         It is possible to generate drifts by changing the mean error rate at change points.
         
         Change points are generated regularly along the stream, based on the number of samples and the number of drifts. 
         A drift is generated every n_samples/n_drifts instances. 
         
         The width of the drifts can be specified. If only one value is given, it is applied to every drift. If several ones are specified,
         they are uniformly reparted and randomly applied to the drifts. 
         
         The new mean error rate after each drift is randomly picked in the mean_errors list.
         
         It is possible to retrive drift chracteristics from the stream : magnitude and severity of drifts. 
         
         Arguments :
            n_samples : int
                Number of instances to generate
             
            n_drifts : int
                Number of drifts to generate
            
            widths_drifts : list of int
                Width(s) to be applied to the drifts. 
                
            mean_errors : list of float
                List of mean errors to simulation the concepts. Must have at least 2 values to simulate drifts from 1 comcept to another.
            
            n_stable_drifts : int, (optional, default = 1)
                Number of floowing drfits with the same patterns (width and mean_errors)
                
            random_state : int (optional, default = 0)
                Random state for the pseudo-random generators. 
        
        """
    def __init__(self,
                 n_samples,
                 n_drifts,
                 widths_drifts,
                 mean_errors,
                 n_stable_drifts = 1,
                 random_state=0):
    
        
        self.n_samples = n_samples
        self.n_drifts = n_drifts
        self.widths_drifts = widths_drifts
        self.mean_errors = mean_errors
        self.n_stable_drifts = n_stable_drifts
        
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.r = np.random.RandomState(self.random_state)
        
        self.stable_period = int(self.n_samples/(self.n_drifts+1))
        
        
        # Generate the list of mean_errors based on the number of drifts and the number of drifts (n_drifts) in a stable period (n_stable_drifts)
        list_err = self.choiceWithoutRepet(math.ceil(self.n_drifts/self.n_stable_drifts)+1,self.mean_errors)
        self.mean_error_concepts = []
        
        if len(self.mean_errors) <= 2 :
            alt = 0
            for d in range(self.n_drifts+1):
                if alt == 0 :
                    self.mean_error_concepts.append(list_err[0])
                    alt = 1
                else :
                    self.mean_error_concepts.append(list_err[1])
                    alt = 0
                
        else :
            ind_err = 0
            alt = 0 
            e1 = list_err[ind_err]
            e2 = list_err[ind_err+1]
            for d in range(self.n_drifts):
                
                if alt == 0 :
                    self.mean_error_concepts.append(e1)
                    alt = 1
                else :
                    self.mean_error_concepts.append(e2)
                    alt = 0
                
                if (((d+1) % self.n_stable_drifts) == 0) and ((d+1) < self.n_drifts) :
                    ind_err += 1
                    e1 = list_err[ind_err]
                    e2 = list_err[ind_err+1]
                    alt = 1
                
            self.mean_error_concepts = np.array(self.mean_error_concepts+[self.mean_error_concepts[-2]])

        self.current_mean_error = None
        self.next_mean_error = None 
        
        # Generate the list of widths based on the number of drifts and the number of drifts (n_drifts) in a stable period (n_stable_drifts)
        if len(self.widths_drifts)>1:
            self.list_widths = np.concatenate([[w]*self.n_stable_drifts for w in self.choiceWithoutRepet(math.ceil(self.n_drifts/self.n_stable_drifts),self.widths_drifts)])
            self.list_widths = self.list_widths[:self.n_drifts]
        else:
            self.list_widths = self.widths_drifts*self.n_drifts
            
        self.current_width = None
        
        self.list_positions = []
        self.position = int(self.n_samples/(self.n_drifts+1))
        self.list_positions = [(drift+1)*self.position for drift in range(self.n_drifts)]
        
        # Manual meta-k base
        self.current_drift_severity = None
        self.current_drift_magnitude = None
        
#        self.error_rates_combin = [list(itertools.combinations(self.mean_errors,2)),
#                                   list(itertools.combinations(self.mean_errors[::-1],2))]
#        self.list_magnitudes = [np.sqrt((np.sqrt(self.error_rates_combin[0][i][0])-np.sqrt(self.error_rates_combin[0][i][1]))**2) for i in range(len(self.error_rates_combin[0]))]
        
        self.name = 'Bernoulli Stream'
        self.n_classes = 2
        
        if len(self.widths_drifts) > self.n_drifts :
            raise ValueError('{} drifts widths specified > number of drifts = {}, should be <'.format(self.widths_drifts,self.n_drifts))
        
            
    #################################################################################################################
       
    @property
    def n_samples(self):
        """ Retrieve the length of the stream.
        Returns
        -------
        int
            The length of the stream.
        """
        return self._n_samples
    
    @n_samples.setter
    def n_samples(self, n_samples):
        """ Set the length of the stream
        Parameters
        ----------
        length of the stream : int
        """
        self._n_samples = n_samples       
    
    @property
    def list_positions(self):
        """ Retrieve the list of drifts positions.
        Returns
        -------
        list
            The list of drifts positions.
        """
        return self._list_positions
    
    @list_positions.setter
    def list_positions(self, list_positions):
        """ Set the list of drifts positions
        Parameters
        ----------
        the list of drifts positions: list
        """
        self._list_positions = list_positions
    
    @property
    def current_mean_error(self):
        """ Retrieve the current_mean_error.
        Returns
        -------
        float
            The current_mean_error.
        """
        return self._current_mean_error
    
    @current_mean_error.setter
    def current_mean_error(self, current_mean_error):
        """ Set the list of current_mean_error
        Parameters
        ----------
        current_mean_error: float
        """
        self._current_mean_error = current_mean_error
    
    @property
    def next_mean_error(self):
        """ Retrieve the next_mean_error.
        Returns
        -------
        float
            The next_mean_error.
        """
        return self._next_mean_error
    
    @next_mean_error.setter
    def next_mean_error(self, next_mean_error):
        """ Set the list of next_mean_error
        Parameters
        ----------
        next_mean_error: float
        """
        self._next_mean_error = next_mean_error
    
    @property
    def current_width(self):
        """ Retrieve the current_width.
        Returns
        -------
        int
            The current_width.
        """
        return self._current_width
    
    @current_width.setter
    def current_width(self, current_width):
        """ Set the list of current_width
        Parameters
        ----------
        current_width: int
        """
        self._current_width = current_width
    
    @property
    def current_drift_severity(self):
        """ Retrieve the current_drift_severity.
        Returns
        -------
        int
            The current_drift_severity.
        """
        return self._current_drift_severity
    
    @current_drift_severity.setter
    def current_drift_severity(self, current_drift_severity):
        """ Set the current_drift_severity
        Parameters
        ----------
        current_drift_severity: int
        """
        self._current_drift_severity = current_drift_severity
    
    @property
    def current_drift_magnitude(self):
        """ Retrieve the current_drift_magnitude.
        Returns
        -------
        int
            The current_drift_magnitude.
        """
        return self._current_drift_magnitude
    
    @current_drift_magnitude.setter
    def current_drift_magnitude(self, current_drift_magnitude):
        """ Set the list of current_drift_magnitude
        Parameters
        ----------
        current_drift_magnitude: float
        """
        self._current_drift_magnitude = current_drift_magnitude
    
    
    #################################################################################################################
    def changeSeed(self, random_state):
        
        self._random_state = check_random_state(random_state)
        self.r = np.random.RandomState(random_state)
        
        
    def prepare_for_use(self):
        """
        Prepares the stream for use.
        Notes
        -----
        This functions should always be called after the stream initialization.
        
        """
        
        self.current_prediction = None
        
        self.sample_idx = 0
        self.concept_idx = 0
        self.width_idx = 0
        
        self.cpt_drifts = 0
        
        self.total_sample_idx = 0
        
        self.current_mean_error = self.mean_error_concepts[self.concept_idx]
        self.next_mean_error = self.mean_error_concepts[self.concept_idx+1]
         
        
        self.current_width = self.list_widths[self.width_idx]
        
        # Manual meta-knowledge
        self.current_drift_severity = self.list_widths[self.cpt_drifts]
        self.current_drift_magnitude = np.sqrt((np.sqrt(self.mean_error_concepts[self.cpt_drifts])-np.sqrt(self.mean_error_concepts[self.cpt_drifts+1]))**2)

#        try :
#            self.current_drift_magnitude = self.list_magnitudes[self.error_rates_combin[0].index((self.current_mean_error,self.next_mean_error))]
#        except ValueError :
#            self.current_drift_magnitude = self.list_magnitudes[self.error_rates_combin[1].index((self.current_mean_error,self.next_mean_error))]

    
    def next_sample(self, batch_size = 1) :
        
        """ next_sample
        The sample generation works as follows: 
            A prediction 0 or 1 is generated by the random Bernoulli process, 
            based on the current and next mean errors. 
            The probability of drift is calculated and updated at every call 
            based on the current sample index and the next drift position and width.
            
            Drift characteristics are udated every time a drift happens. 
            
        Parameters
        ----------
        batch_size: int
            The number of samples to return (works for batch_size == 1 only for the moment)
        Returns
        -------
        tuple or tuple list
            Return a tuple with the predictions matrix for
            the batch_size samples that were requested.
        """
        
        self.current_prediction = np.zeros(batch_size)
        
        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / float(self.current_width)
            probability_drift = 1.0 / (1.0 + np.exp(x))
            
            if self._random_state.rand() > probability_drift:
                pred = self.perform_bernoulli_trials(1,1-self.current_mean_error)
            else:
                pred = self.perform_bernoulli_trials(1,1-self.next_mean_error)
                
            self.current_prediction[j] = pred
            
            # Update theoritical meta-features at every exact drift point
            if self.total_sample_idx in self.list_positions:
                # Manual meta-knowledge
                self.current_drift_severity = self.list_widths[self.cpt_drifts]
                self.current_drift_magnitude = np.sqrt((np.sqrt(self.mean_error_concepts[self.cpt_drifts])-np.sqrt(self.mean_error_concepts[self.cpt_drifts+1]))**2)
                self.cpt_drifts +=1
                
            if (self.sample_idx >= self.stable_period + self.current_width) and (self.concept_idx<self.n_drifts-1):
                self.sample_idx = 0
                
                self.width_idx += 1
                self.concept_idx += 1
                
                self.current_mean_error = self.mean_error_concepts[self.concept_idx]
                self.next_mean_error = self.mean_error_concepts[self.concept_idx+1]
                
                self.current_width = self.list_widths[self.width_idx]
                
#                try :
#                    self.current_drift_magnitude = self.list_magnitudes[self.error_rates_combin[0].index((self.current_mean_error,self.next_mean_error))]
#                except ValueError :
#                    self.current_drift_magnitude = self.list_magnitudes[self.error_rates_combin[1].index((self.current_mean_error,self.next_mean_error))]
#                

            elif (self.sample_idx >= self.stable_period + self.current_width) and (self.concept_idx>=self.n_drifts-1) : 
                self.sample_idx = 0
                self.concept_idx += 1
                self.current_mean_error = self.mean_error_concepts[self.concept_idx]
                
            
        self.total_sample_idx += batch_size
        
        return self.current_prediction
    
    def perform_bernoulli_trials(self, n, p):
        """
            Perform n Bernoulli trials with success probability p
            and return number of successes.
        """
        # Initialize number of successes: n_success
        n_success = 0
    
        # Perform trials
        for i in range(n):
            # Choose random number between zero and one: random_number
            random_number = self.r.rand()
    
            # If less than p, it's a success so add one to n_success
            if random_number < p:
                n_success += 1
    
        return n_success

    def choiceWithoutRepet(self, n_iter, list_choices):
        """
            Generate a list of n_iter items from lit_choices without following repetition.
            If len(list_choice) > 2, the list is generated without repetitions every 3 items.
        """
        final_list = []
        
        if len(list_choices) <= 2 :
            for i in range(n_iter) :
                if i == 0 :
                    final_list.append(self.r.choice(list_choices, 1)[0])
                else :
                    value = self.r.choice(list_choices, 1)[0]
                    while value == final_list[i-1]:
                        value = self.r.choice(list_choices, 1)[0]
                    final_list.append(value)
        
        else :
             for i in range(n_iter) :
                if i == 0 :
                    final_list.append(self.r.choice(list_choices, 1)[0])
                else :
                    value = self.r.choice(list_choices, 1)[0]
                    while (value == final_list[i-1]) or (value == final_list[i-2]):
                        value = self.r.choice(list_choices, 1)[0]
                    final_list.append(value)
        
            
        return final_list
    
            
    
    def get_data_info(self):
        """ Retrieves information from the stream
         
         The default format is: 'Stream name - n_samples, n_drifts, widths_drifts, error_rates'.
         
         Returns
        -------
        string
            Stream data information
        
        """
        
        return self.name + " - {} samples(s), {} drifts, widths_drifts : {}, mean_errors: {}".format(self.n_samples, self.n_drifts,
                                                                           self.widths_drifts, self.mean_errors)


####################################################################
## Test
#from scipy.stats import kurtosis, skew
#     
#from scipy.spatial import distance
#
#mag_theo = [0.632,0.578,0.447,0.389,0.289,0.227,0.142,0.067]
#sev_theo = [1,100,500]
#
#meta_features = list(itertools.product(sev_theo,mag_theo))
#
#n_smpl = 500000
#stream = BernouilliStream(n_samples=n_smpl, n_drifts = 500, widths_drifts = [1,100,500], mean_errors = [0.1,0.3,0.6,0.9], n_stable_drifts = 6)
#stream.prepare_for_use()
#
#drift_positions = stream.list_positions
#

    

