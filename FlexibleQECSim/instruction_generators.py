from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional, Iterable,Dict
import stim
import numpy as np
from abc import ABC, abstractmethod
from itertools import cycle, islice

Etype_to_stim_target_fun = {
    'X': stim.target_x,
    'Y': stim.target_y,
    'Z': stim.target_z,
}

def chunked(iterable, chunk_size):
    """Yield successive n-sized chunks from the input iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


@dataclass
class SQE: #stands for Single_qubit_error/event
    """
    Determines what's the type of error on the data qubit (I,X,Y,Z) and if it's heralded
    """
    type: str
    heralded: bool
    def __post_init__(self):
        assert self.type in ['I','X','Y','Z'] 

@dataclass
class MQE: #stands for Multi_qubit_error/event
    """
    Constitutes one of the many disjoint probabilities in an Error_mechanism
    """
    p: float
    list_of_SQE: List[SQE]
    def __post_init__(self):
        assert self.p >= 0, "can't create an event with below 0 probability?"

@dataclass
class InsGenerator(ABC): # "Ins" stands for (stim) "Instruction"
    list_of_MQE: List[MQE]
    # generator_type: str = 'Base generator'
    def __post_init__(self):
        sum_of_prob = sum([mqe.p for mqe in self.list_of_MQE])
        assert  sum_of_prob > 1-1e-7 and sum_of_prob < 1+ 1e-7
        self.num_qubits = len(self.list_of_MQE[0].list_of_SQE) # the number of data qubits it describes
        assert all(len(mqe.list_of_SQE) == self.num_qubits for mqe in self.list_of_MQE) # Ensure the error model describes errors on a fixed amount of data qubits

    @abstractmethod
    def get_instruction(self, qubits:List[int]) -> List:
        pass   

    def __repr__(self):
        return f'{__class__.__name__} of type {self.generator_type}'

@dataclass
class InsGeneratorStepwiseProbs(InsGenerator):
    # generator_type: str = 'StepwiseProbs base generator'
    def __post_init__(self):
        # Convert the probabilities to those used in CORRELATED_ERROR and ELSE_CORRELATED_ERRORs (used for non-vectorized methods only, which are Deprecated)
        self.list_of_MQE = sorted(self.list_of_MQE,reverse=True, key=lambda x: x.p)
        self.stepwise_probabilities = []
        prob_left = 1
        for event in self.list_of_MQE:
            stepwise_p = event.p / prob_left
            stepwise_p= max(min(stepwise_p, 1), 0)
            self.stepwise_probabilities.append(stepwise_p)
            prob_left -= event.p

@dataclass
class InsGeneratorConditional(InsGenerator):
    herald_locations: Optional[List[int]] = field(init=False,default=None)
    num_herald_locations: Optional[int] = field(init=False,default=None)
    
    def get_padded_new_ancillas_array_update_list(self,data_qubits_array,index_in_list):
        num_parallel = data_qubits_array.shape[1]
        num_ancillas= num_parallel * self.num_herald_locations
        
        padded_ancillas = np.zeros(data_qubits_array.shape, dtype=int)
        counter = index_in_list[0]
        fill_values = np.arange(counter, counter + num_ancillas)
        fill_values = fill_values.reshape(num_parallel, self.num_herald_locations).T
        padded_ancillas[self.herald_locations, :] = fill_values
        index_in_list[0] += num_ancillas
        return padded_ancillas, num_parallel
    
@dataclass
class InsGeneratorPosteriorProbs(InsGeneratorConditional):
    generator_type: str = 'PosteriorProbs base generator'
    def __post_init__(self):
        # Prepare conditional probabilities for decoding erasure detection 
        # compute the arithmatic sum of I/X/Y/Z error rates over num_qubits qubits (using the sum without heralding in normal circuit is the same as using the *disjoint* components without heralding)
        #   also compute the sum of heralded pauli errors for num_qubits qubits
        self.Etype_to_sum = [None] * self.num_qubits
        self.Etype_to_heralded_sum = [None] * self.num_qubits
        self.conditional_probabilities = [None] * self.num_qubits
        self.p_herald = [0] * self.num_qubits
        for i in range(self.num_qubits):
            self.Etype_to_sum[i] = {
                'I':0,
                'X':0,
                'Y':0,
                'Z':0
            }
            self.Etype_to_heralded_sum[i] = {
                'I':0,
                'X':0,
                'Y':0,
                'Z':0
            }
            for mqe in self.list_of_MQE:
                Etype_on_this_qubit = mqe.list_of_SQE[i].type
                self.Etype_to_sum[i][Etype_on_this_qubit] += mqe.p
                if mqe.list_of_SQE[i].heralded == True:
                    self.Etype_to_heralded_sum[i][Etype_on_this_qubit] += mqe.p

            # Compute the conditional probabilities
            self.p_herald[i] = sum(self.Etype_to_heralded_sum[i].values())
            if self.p_herald[i] != 0:
                self.conditional_probabilities[i] = [
                    self.Etype_to_heralded_sum[i]['X']/self.p_herald[i], 
                    self.Etype_to_heralded_sum[i]['Y']/self.p_herald[i], 
                    self.Etype_to_heralded_sum[i]['Z']/self.p_herald[i],
                    (self.Etype_to_sum[i]['X']-self.Etype_to_heralded_sum[i]['X'])/(1-self.p_herald[i]),
                    (self.Etype_to_sum[i]['Y']-self.Etype_to_heralded_sum[i]['Y'])/(1-self.p_herald[i]),
                    (self.Etype_to_sum[i]['Z']-self.Etype_to_heralded_sum[i]['Z'])/(1-self.p_herald[i])
                ]
        heralded_locations_one_hot_encoding  = [prob > 0 for prob in self.p_herald]
        self.herald_locations = np.where(heralded_locations_one_hot_encoding)[0]
        self.num_herald_locations = np.sum(heralded_locations_one_hot_encoding)


@dataclass
class DummyInsGenerator(InsGenerator):
    generator_type: str = 'Dummy generator'
    def __post_init__(self):
        self.num_qubit_called = 0

    def get_instruction(self, qubits:List[int]) -> List:
        self.num_qubit_called += len(qubits)
        return ([
            ['X_ERROR',qubits,0.5]
        ])
        
@dataclass
class NormalInsGenerator(InsGeneratorStepwiseProbs):
    
    instruction_name: Optional[str] = field(default=None)
    instruction_arg: Union[float, List[float]] = field(default=None)
    vectorizable: Optional[bool] = field(default=None)
    generator_type: str = 'Normal generator'
    def __post_init__(self):
        InsGenerator.__post_init__(self)
        if  self.instruction_name == None or self.instruction_arg == None:
            self.vectorizable = False
            InsGeneratorStepwiseProbs.__post_init__(self)
        else:
            self.vectorizable = True

    def get_instruction(self, qubits:List[int]) -> List:
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        if self.vectorizable:
            list_of_args = []
            list_of_args.append([self.instruction_name,qubits,self.instruction_arg])
            return list_of_args
        else:
            list_of_args_gathered = []
            for sets_of_qubits in chunked(iterable=qubits, chunk_size = self.num_qubits):
                list_of_args = []
                for mqe, stepwise_p in zip(self.list_of_MQE,self.stepwise_probabilities):
                    targets = []
                    for sqe,qubit in zip(mqe.list_of_SQE, sets_of_qubits):# the len of qubits can be smaller than len of event.list_of_Etype, but the smallest len count will determin how many iterations are run in zip()
                        if sqe.type != 'I':
                            targets.append(Etype_to_stim_target_fun[sqe.type](qubit))
                    list_of_args.append(["ELSE_CORRELATED_ERROR",targets,stepwise_p])
                list_of_args[0][0] = "CORRELATED_ERROR"
                list_of_args_gathered.extend(list_of_args)
            return list_of_args_gathered

@dataclass
class ErasureInsGenerator(InsGeneratorStepwiseProbs, InsGeneratorPosteriorProbs):
    '''
        Whether an erasure instruction is vectorizable is dependent on whether there's correlation between two data qubits.
        If during a 2-qubit gate, the two qubits are independently erased, then we apply two pairs of PAULI_CHANNEL_2.
        But if the 2-qubit gate is described by some correlated erasure error, as in PHYSICAL REVIEW X 13, 041013 (2023), 
            then the mechanism involving more than 2 qubits need to be modeled by CORRELATED_ERROR, and CORRELATED_ERROR are not vectorizable.

        instruction_name and instruction_arg can be used to describe one PAULI_CHANNEL_2 that is applied to one (pair (with broadcasting)) of qubits
            or two PAULI_CHANNEL_2 that is separatly applied to a pair of qubits
    '''
    instruction_name: Optional[Union[str, List[str]]] = field(default=None)
    instruction_arg: Optional[Union[float, List[float],List[List[float]]]] = field(default=None)
    vectorizable: Optional[bool] = field(default=None)
    generator_type: str = 'Dummy generator'
    def __post_init__(self):
        InsGenerator.__post_init__(self)
        if self.instruction_name == None or self.instruction_arg == None:
            self.vectorizable = False
            InsGeneratorStepwiseProbs.__post_init__(self)
        else:
            self.vectorizable = True
            InsGeneratorPosteriorProbs.__post_init__(self)
            if isinstance(self.instruction_name,List):
                assert len(self.instruction_name) == len(self.instruction_arg)
            else:
                self.instruction_name = [self.instruction_name]
                self.instruction_arg = [self.instruction_arg]
            while len(self.instruction_name) < self.num_herald_locations:
                self.instruction_name.append(self.instruction_name[-1])
                self.instruction_arg.append(self.instruction_arg[-1])
            
    def get_instruction(self, 
                        qubits:List[int],
                        next_ancilla_qubit_index_in_list:List[int]) -> List:
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        data_qubits_array = np.array(qubits).reshape(-1,self.num_qubits).T

        padded_ancillas, num_parallel = self.get_padded_new_ancillas_array_update_list(data_qubits_array,next_ancilla_qubit_index_in_list)

        if self.vectorizable:
            list_of_args = []
            for i in range(self.num_qubits):
                if i in self.herald_locations: 
                    interposed_array = np.empty(num_parallel*2, dtype=int)
                    interposed_array[::2] = data_qubits_array[i]
                    interposed_array[1::2] = padded_ancillas[i]
                    list_of_args.append([self.instruction_name[i], interposed_array, self.instruction_arg[i]])
            return list_of_args
        else:
            ancillas = ancillas.T.flatten()
            ancilla_idx = 0
            list_of_args_gathered = []
            for sets_of_qubits in chunked(iterable=qubits, chunk_size = self.num_qubits):
                list_of_args = []
                for mqe, stepwise_p in zip(self.list_of_MQE,self.stepwise_probabilities):
                    targets = []
                    for sqe,circuit_qubit_idx in zip(mqe.list_of_SQE, sets_of_qubits):
                        if sqe.type != 'I':
                            targets.append(Etype_to_stim_target_fun[sqe.type](circuit_qubit_idx))
                        if sqe.heralded:# If it heralded, then the corresponding ancilla must have been assigned on this qubit for this Gate_error_model 
                            targets.append(Etype_to_stim_target_fun['X'](ancillas[ancilla_idx]))
                            ancilla_idx += 1
                    list_of_args.append(["ELSE_CORRELATED_ERROR",targets,stepwise_p])
                list_of_args[0][0] = "CORRELATED_ERROR" # The first instruction should be CORRELATED_ERROR in stim
                list_of_args_gathered.extend(list_of_args)
            return list_of_args_gathered   

@dataclass
class PosteriorInsGenerator(InsGeneratorPosteriorProbs):
    generator_type: str = 'Posterior generator'
    def __post_init__(self):
        InsGenerator.__post_init__(self)
        InsGeneratorPosteriorProbs.__post_init__(self)

    def get_instruction(self,qubits:List[int],
                        erasure_measurement_index_in_list:List[int],
                        single_measurement_sample:Union[List,np.array]) -> List:
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        data_qubits_array = np.array(qubits).reshape(-1,self.num_qubits).T
        
        if self.num_herald_locations > 0:
            padded_ancillas,num_parallel = self.get_padded_new_ancillas_array_update_list(data_qubits_array,erasure_measurement_index_in_list)

            # Get bool array signifing erasure detection. Note that erasure_meas has length num_qubits instead of num_herald_locations
            erasure_meas = np.zeros(padded_ancillas.shape, dtype=bool)
            erasure_meas[self.herald_locations, :] = single_measurement_sample[padded_ancillas]
            
            # For each qubit location, get two conditional probabilities array or the static probability if erasure conversion not used. 
            list_of_args = []
            for i in range(self.num_qubits): # This looks like a loop, but this loop is at most length-2. It's still parrallized. Note that the posterior probabilities are applied as single qubit errors.
                if i in self.herald_locations: 
                    converted_data_qubits = data_qubits_array[i][np.where(erasure_meas[i])[0]] #data_qubits_array[i] is a 1-d array
                    no_detection_data_qubits = data_qubits_array[i][np.where(~erasure_meas[i])[0]]
                    list_of_args.append(["PAULI_CHANNEL_1", converted_data_qubits, [self.conditional_probabilities[i][0], self.conditional_probabilities[i][1], self.conditional_probabilities[i][2]]])
                    list_of_args.append(["PAULI_CHANNEL_1", no_detection_data_qubits, [self.conditional_probabilities[i][3], self.conditional_probabilities[i][4], self.conditional_probabilities[i][5]]])
                else:
                    list_of_args.append(["PAULI_CHANNEL_1", data_qubits_array[i], [self.Etype_to_sum[i]['X'], self.Etype_to_sum[i]['Y'], self.Etype_to_sum[i]['Z']]])
        else:
            list_of_args = []
            for i in range(self.num_qubits):
                list_of_args.append(["PAULI_CHANNEL_1", data_qubits_array[i], [self.Etype_to_sum[i]['X'], self.Etype_to_sum[i]['Y'], self.Etype_to_sum[i]['Z']]])
        return list_of_args   
    
@dataclass
class DeterministicInsGenerator(InsGeneratorConditional):
    '''
    It can handle both 
    DeterministicInsGenerator deterministically insert instruction based on a pre-rolled dice_index_in_list,
        this makes is similar to PosteriorInsGenerator.
    DeterministicInsGenerator also applies instruction to both the data qubits and virtual-erasure-flag-ancillas,
        this makes it similar to ErasureInsGenerator
    '''
    
    num_dice: int
    instruction_name: Union[str, List[str]]
    instruction_arg: Union[float, List[float],List[List[float]]]

    num_qubit_per_dice: Optional[int]= field(default=None)
    generator_type: str = 'Deterministic generator'

    herald_locations: List[int] = field(init=False,default=None)
    num_herald_locations: int = field(init=False,default=None)

    def __post_init__(self):
        InsGenerator.__post_init__(self)

        self.num_herald_locations = self.num_dice
        self.herald_locations = np.array(range(self.num_dice))
        self.num_qubit_per_dice = int(self.num_qubits/self.num_dice)


        if isinstance(self.instruction_name,List):
            assert len(self.instruction_name) == len(self.instruction_arg)
        else:
            self.instruction_name = [self.instruction_name]
            self.instruction_arg = [self.instruction_arg]

        while len(self.instruction_name) < self.num_herald_locations:
            self.instruction_name.append(self.instruction_name[-1])
            self.instruction_arg.append(self.instruction_arg[-1])

        InsGeneratorConditional.__post_init__(self)

    def get_instruction(self,qubits:List[int],
                        dice_index_in_list:List[int],
                        single_dice_sample:Union[List,np.array]) -> List:
        '''
        I don't check if self.num_herald_locations > 0 because if self.num_herald_locations == 0 it must be trivial mechanism
        '''
        assert len(qubits) % self.num_qubits == 0, "wrong number of qubits"
        data_qubits_array = np.array(qubits).reshape(-1,self.num_qubits).T
        padded_dices,num_parallel = self.get_padded_new_ancillas_array_update_list(data_qubits_array,dice_index_in_list) # padded_dices is an array of dice idx
        
        positive_dices = np.zeros(padded_dices.shape, dtype=bool) #positive_dices is an array of dice results corresponding to padded_dices
        positive_dices[self.herald_locations, :] = single_dice_sample[padded_dices[self.herald_locations, :]]
        
        list_of_args = []
        for i in range(self.num_dice):
            qubit_chunk = data_qubits_array[i*self.num_qubit_per_dice : (i+1)*self.num_qubit_per_dice] #qubit_chunk is 2-d array, row is qubit location, column is parralization at a single time slice
            positive_locations = qubit_chunk[:,np.where(positive_dices[i])[0]] # 2-d array containing qubit indices,row is qubit location, length is num_qubit_per_dice, column is parralization at a single time slice, 
            targets = positive_locations.T.flatten()
            if len(targets)>0:
                list_of_args.append([self.instruction_name[i], targets, self.instruction_arg[i]])

        return list_of_args   