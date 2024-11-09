from FlexibleQECSim.instruction_generators import *



@dataclass
class ErrorMechanism:
    """
    A ErrorMechanism constitutes multiple MQEs, can describe the intrinsic error or the erasure conversion of the gate.
    This class specify some error events with *disjoint* probabilities. 
    The MQEs are disjoint because it's good enough approximation to neglect events where multiple MQEs happening at the same time.
    It also specify which error events can be heralded, and herald rate.
    Vectorization:
        different modes have different vectorization rules.
        | Type              | normal mode               |   erasure mode                |   dynamic mode            |   deterministic mode      |
        |-------------------|---------------------------|-------------------------------|---------------------------|---------------------------|
        |1-q nonherald      |Error,q,param              |Error,q,param                  |Error,q,param              |Error,q,param'             |
        |1-q herald         |Error,q,param              |PAULI_CHANNEL_2,[q,a],param    |PAULI_CHANNEL_1,q,param'   |PAULI_CHANNEL_2,q,param'   |
        |2-q nonherald      |Error,[q,p],param          |Error,[q,p],param              |Error,[q,p],param          |Error,[q,p],param'         |
        |2-q herald         |Error,[q,p],param          |PAULI_CHANNEL_2,[q,a],param *2 |PAULI_CHANNEL_1,q,param'*2 |PAULI_CHANNEL_2,[q,p],param'| (there's no heralded 2-qubit errors, decompose them into 1-q heralds)
    """
    normal_generator: NormalInsGenerator
    erasure_generator: Optional[ErasureInsGenerator] = None
    posterior_generator: Optional[PosteriorInsGenerator] = None
    deterministic_generator: Optional[DeterministicInsGenerator] = None
    dummy_generator: Optional[DummyInsGenerator] = None

    next_ancilla_qubit_index_in_list: Optional[int] = None

    erasure_measurement_index_in_list: Optional[int] = None
    single_measurement_sample:  Optional[Union[List,np.array]] = None

    next_dice_index_in_list: Optional[int] = None
    single_dice_sample:  Optional[Union[List,np.array]] = None

    name:Optional[str] = None

    def __post_init__(self):
        self.num_qubits = self.normal_generator.num_qubits
        self.dummy_generator = DummyInsGenerator(list_of_MQE=[])

        if self.posterior_generator is None or self.posterior_generator.num_herald_locations == 0:
            self.is_erasure = False
        else:
            self.is_erasure = True
        if self.erasure_generator is None or self.posterior_generator is None:
            self.erasure_generator = self.normal_generator
            self.posterior_generator = self.normal_generator

    def get_instruction(self, 
                        qubits: Union[List[int], Tuple[int]],
                        mode:str):
        '''
        return list of args that can be used in  stim.circuit.append()
        '''


        if mode == 'deterministic':
            instructions =  self.deterministic_generator.get_instruction(qubits,self.next_dice_index_in_list,self.single_dice_sample)
        elif mode == 'dummy':
            instructions =  self.dummy_generator.get_instruction(qubits)
        elif mode == 'normal' or self.is_erasure == False: 
            instructions = self.normal_generator.get_instruction(qubits=qubits)
        elif mode == 'erasure':
            instructions =  self.erasure_generator.get_instruction(qubits,self.next_ancilla_qubit_index_in_list)
        elif mode == 'posterior':
            instructions =  self.posterior_generator.get_instruction(qubits,self.erasure_measurement_index_in_list,self.single_measurement_sample)

        else:
            raise Exception("unsupported mode")
        
        # with open('log.txt', 'a') as file:
        #     # Write the instructions to the file, starting on a new line
        #     file.write('\n name:'+self.name+'  mode:'+mode+'\n')
        #     file.write(str(instructions))

        return instructions
    def __repr__(self) -> str:
        return self.name

@dataclass
class GateErrorModel:
    """
    A GateErrorModel contains one or more ErrorMechanism,
    Different ErrorMechanism within a GateErrorModel are considered independent (they can happen at the same time)
    A GateErrorModel can only have one erasure conversion mechanism.
    """

    list_of_mechanisms: List[ErrorMechanism]
    name_to_mechanism: Dict[str, ErrorMechanism] = field(init=False)

    def __post_init__(self):
        self.name_to_mechanism = {mechanism.name: mechanism for mechanism in self.list_of_mechanisms}
        if len(self.list_of_mechanisms) == 0:
            self.trivial = True
            return
        else:
            self.trivial = False
            
        assert all([mechanism.num_qubits == self.list_of_mechanisms[0].num_qubits for mechanism in self.list_of_mechanisms])
        assert sum([mechanism.is_erasure for mechanism in self.list_of_mechanisms]) <= 1

    
    def set_next_ancilla_qubit_index_in_list(self,next_ancilla_qubit_index_in_list: int):
        for mechanism in self.list_of_mechanisms:
            mechanism.next_ancilla_qubit_index_in_list = next_ancilla_qubit_index_in_list
    def set_erasure_measurement_index_in_list(self,erasure_measurement_index_in_list: int):
        for mechanism in self.list_of_mechanisms:
            mechanism.erasure_measurement_index_in_list = erasure_measurement_index_in_list
    def set_single_measurement_sample(self,single_measurement_sample: Union[List,np.array]):
        for mechanism in self.list_of_mechanisms:
            mechanism.single_measurement_sample = single_measurement_sample
    
    def get_instruction(self, 
                        qubits: Union[List[int], Tuple[int]],
                        mode:str):
        if self.trivial:
            return []
        list_of_args = []
        for mechanism in self.list_of_mechanisms:
            list_of_args += mechanism.get_instruction(qubits=qubits,mode=mode)
        return list_of_args
    
    
    def __repr__(self) -> str:
        s = '''Gate_error_model of the following mechanisms:\n'''
        for i,mech in enumerate(self.list_of_mechanisms):
            s += f"mechanism {i}, name:{mech.name} \n"
            # s += mech.__repr__()
            
        return  s




def get_1q_depolarization_mechanism(p_p):
    list_of_MQE=  [   
                MQE(1-p_p,[SQE("I",False)]),
                MQE(p_p/3,[SQE("X",False)]),
                MQE(p_p/3,[SQE("Y",False)]),
                MQE(p_p/3,[SQE("Z",False)])
            ]
    normal_generator = NormalInsGenerator(
            list_of_MQE = list_of_MQE,
            instruction_name ='DEPOLARIZE1',
            instruction_arg = p_p)
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 1,
            instruction_name = 'DEPOLARIZE1',
            instruction_arg = 1.0,
        )
    return ErrorMechanism(
        normal_generator = normal_generator,
        deterministic_generator = deterministic_generator,
        name = '1q depo'
        )

def get_1q_differential_shift_mechanism(p_z_shift):
    list_of_MQE= [   
            MQE(1-p_z_shift,[SQE("I",False)]),
            MQE(p_z_shift,[SQE("Z",False)])
        ]
    normal_generator = NormalInsGenerator(
            list_of_MQE = list_of_MQE,
            instruction_name ='Z_ERROR',
            instruction_arg = p_z_shift)
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 1,
            instruction_name = 'Z_ERROR',
            instruction_arg = 1.0,
        )
    return ErrorMechanism(
        normal_generator= normal_generator,
        deterministic_generator = deterministic_generator,
        name = '1q z shift'
        )

def get_1q_biased_erasure_mechanism(p_e):
    list_of_MQE=  [   
            MQE(1 - p_e,[SQE("I",False)]),
            MQE(p_e/2,[SQE("I",True)]),
            MQE(p_e/2,[SQE("Z",True)])
        ]
    normal_generator = NormalInsGenerator(
            list_of_MQE = list_of_MQE,
            instruction_name ='Z_ERROR',
            instruction_arg = p_e/2)
    erasure_generator = ErasureInsGenerator(
            list_of_MQE = list_of_MQE,
            instruction_name = "PAULI_CHANNEL_2",
            instruction_arg = [
                  # ix iy iz
                    p_e / 2, 0, 0,
                  # xi xx xy xz
                    0, 0, 0, 0,
                  # yi yx yy yz
                    0, 0, 0, 0,
                  # zi zx zy zz
                    0, p_e / 2, 0, 0
                ])
    posterior_generator = PosteriorInsGenerator(
            list_of_MQE = list_of_MQE
            )
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 1,
            instruction_name = 'Z_ERROR',
            instruction_arg = 0.5,
        )
    return ErrorMechanism(
        normal_generator = normal_generator,
        erasure_generator=erasure_generator,
        posterior_generator = posterior_generator,
        deterministic_generator = deterministic_generator,
        name = '1q erasure'
        )

def get_1q_error_model(p_e,p_z_shift, p_p):
    mechanism_list = [get_1q_depolarization_mechanism(p_p)]
    if p_z_shift>0:
        mechanism_list.append(get_1q_differential_shift_mechanism(p_z_shift))
    if p_e>0:
        mechanism_list.append(get_1q_biased_erasure_mechanism(p_e))
    return GateErrorModel(mechanism_list)




def get_2q_depolarization_mechanism(p_p):
    list_of_MQE= [
           MQE(1-p_p,[SQE("I",False),SQE("I",False)]),

            MQE(p_p/15,[SQE("I",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("I",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("I",False),SQE("Z",False)]),

            MQE(p_p/15,[SQE("X",False),SQE("I",False)]),
            MQE(p_p/15,[SQE("X",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("X",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("X",False),SQE("Z",False)]),

            MQE(p_p/15,[SQE("Y",False),SQE("I",False)]),
            MQE(p_p/15,[SQE("Y",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("Y",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("Y",False),SQE("Z",False)]),

            MQE(p_p/15,[SQE("Z",False),SQE("I",False)]),
            MQE(p_p/15,[SQE("Z",False),SQE("X",False)]),
            MQE(p_p/15,[SQE("Z",False),SQE("Y",False)]),
            MQE(p_p/15,[SQE("Z",False),SQE("Z",False)]),
        ]
    normal_generator = NormalInsGenerator(
            list_of_MQE = list_of_MQE,
            instruction_name ='DEPOLARIZE2',
            instruction_arg = p_p)
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 1,
            instruction_name = 'DEPOLARIZE2',
            instruction_arg = 1,
        )
    return ErrorMechanism(
        normal_generator=normal_generator,
        deterministic_generator = deterministic_generator,
        name = '2q depo'
        )

def get_2q_differential_shift_mechanism(p_z_shift):
    list_of_MQE= [
            MQE((1-p_z_shift)**2,[SQE("I",False),SQE("I",False)]),
            MQE(p_z_shift * (1-p_z_shift),[SQE("Z",False),SQE("I",False)]),
            MQE(p_z_shift * (1-p_z_shift),[SQE("I",False),SQE("Z",False)]),
            MQE(p_z_shift**2,[SQE("Z",False),SQE("Z",False)]),
        ]
    normal_generator = NormalInsGenerator(
        list_of_MQE = list_of_MQE,
        instruction_name ='Z_ERROR',
        instruction_arg = p_z_shift)
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 2,
            instruction_name = 'Z_ERROR',
            instruction_arg = 1.0,
        )
    return ErrorMechanism(
        normal_generator= normal_generator,
        deterministic_generator = deterministic_generator,
        name = '2q z shift'
        )

def get_2q_biased_erasure_mechanism(p_e):
    list_of_MQE=  [   
                MQE((1- p_e)**2,[SQE("I",False),SQE("I",False)]), # no detection cases

                MQE(p_e/2 * (1- p_e),[SQE("I",True),SQE("I",False)]), # Single qubit detection cases
                MQE(p_e/2 * (1- p_e),[SQE("Z",True),SQE("I",False)]),
                MQE((1- p_e) * p_e/2,[SQE("I",False),SQE("I",True)]),
                MQE((1- p_e) * p_e/2,[SQE("I",False),SQE("Z",True)]),

                MQE( (p_e/2)**2,[SQE("I",True),SQE("I",True)]), # Two qubit detection cases
                MQE( (p_e/2)**2,[SQE("I",True),SQE("Z",True)]),
                MQE( (p_e/2)**2,[SQE("Z",True),SQE("I",True)]),
                MQE( (p_e/2)**2,[SQE("Z",True),SQE("Z",True)]),
            ]
    normal_generator = NormalInsGenerator(
        list_of_MQE = list_of_MQE,
        instruction_name ='Z_ERROR',
        instruction_arg = p_e/2)
    erasure_generator = ErasureInsGenerator(
        list_of_MQE = list_of_MQE,
        instruction_name = "PAULI_CHANNEL_2",
        instruction_arg = [
                # ix iy iz
                p_e / 2, 0, 0,
                # xi xx xy xz
                0, 0, 0, 0,
                # yi yx yy yz
                0, 0, 0, 0,
                # zi zx zy zz
                0, p_e / 2, 0, 0
            ])
    posterior_generator = PosteriorInsGenerator(
            list_of_MQE = list_of_MQE
            )
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 2,
            instruction_name = 'Z_ERROR',
            instruction_arg = 0.5,
        )
    return ErrorMechanism(
        normal_generator = normal_generator,
        erasure_generator=erasure_generator,
        posterior_generator = posterior_generator,
        deterministic_generator = deterministic_generator,
        name = '2q erasure'
        )

def get_2q_erasure_mechanism(p_e):
    list_of_MQE=  [   
                MQE((1- p_e)**2,[SQE("I",False),SQE("I",False)]), # no detection cases

                MQE(p_e/4 * (1- p_e),[SQE("I",True),SQE("I",False)]), # Single qubit detection cases
                MQE(p_e/4 * (1- p_e),[SQE("X",True),SQE("I",False)]),
                MQE(p_e/4 * (1- p_e),[SQE("Y",True),SQE("I",False)]),
                MQE(p_e/4 * (1- p_e),[SQE("Z",True),SQE("I",False)]),

                MQE((1- p_e) * p_e/4,[SQE("I",False),SQE("I",True)]),
                MQE((1- p_e) * p_e/4,[SQE("I",False),SQE("X",True)]),
                MQE((1- p_e) * p_e/4,[SQE("I",False),SQE("Y",True)]),
                MQE((1- p_e) * p_e/4,[SQE("I",False),SQE("Z",True)]),

                MQE( (p_e/4)**2,[SQE("I",True),SQE("I",True)]), # Two qubit detection cases
                MQE( (p_e/4)**2,[SQE("I",True),SQE("X",True)]),
                MQE( (p_e/4)**2,[SQE("I",True),SQE("Y",True)]),
                MQE( (p_e/4)**2,[SQE("I",True),SQE("Z",True)]),
                
                MQE( (p_e/4)**2,[SQE("X",True),SQE("I",True)]),
                MQE( (p_e/4)**2,[SQE("X",True),SQE("X",True)]),
                MQE( (p_e/4)**2,[SQE("X",True),SQE("Y",True)]),
                MQE( (p_e/4)**2,[SQE("X",True),SQE("Z",True)]),

                MQE( (p_e/4)**2,[SQE("Y",True),SQE("I",True)]),
                MQE( (p_e/4)**2,[SQE("Y",True),SQE("X",True)]),
                MQE( (p_e/4)**2,[SQE("Y",True),SQE("Y",True)]),
                MQE( (p_e/4)**2,[SQE("Y",True),SQE("Z",True)]),

                MQE( (p_e/4)**2,[SQE("Z",True),SQE("I",True)]),
                MQE( (p_e/4)**2,[SQE("Z",True),SQE("X",True)]),
                MQE( (p_e/4)**2,[SQE("Z",True),SQE("Y",True)]),
                MQE( (p_e/4)**2,[SQE("Z",True),SQE("Z",True)]),
            ]
    normal_generator = NormalInsGenerator( # This is per-qubit
        list_of_MQE = list_of_MQE,
        instruction_name ='DEPOLARIZE1',
        instruction_arg = p_e*3/4)
    erasure_generator = ErasureInsGenerator( # This is per-qubit
        list_of_MQE = list_of_MQE,
        instruction_name = "PAULI_CHANNEL_2",
        instruction_arg = [
                # ix iy iz
                p_e / 4, 0, 0,
                # xi xx xy xz
                0, p_e / 4, 0, 0,
                # yi yx yy yz
                0, p_e / 4, 0, 0,
                # zi zx zy zz
                0, p_e / 4, 0, 0
            ])
    posterior_generator = PosteriorInsGenerator(
            list_of_MQE = list_of_MQE
            )
    deterministic_generator = DeterministicInsGenerator(
            list_of_MQE = list_of_MQE,
            num_dice = 2,
            instruction_name = 'DEPOLARIZE1',
            instruction_arg = p_e*3/4,
        )
    return ErrorMechanism(
        normal_generator = normal_generator,
        erasure_generator=erasure_generator,
        posterior_generator = posterior_generator,
        deterministic_generator = deterministic_generator,
        name = '2q erasure unbiased'
        )

def get_2q_error_model(p_p,
                       p_e,
                       p_z_shift = 0,
                       biased=True):
    mechanism_list = [get_2q_depolarization_mechanism(p_p)]
    if p_z_shift>0:
        mechanism_list.append(get_2q_differential_shift_mechanism(p_z_shift))
    if p_e>0:
        if biased:
            mechanism_list.append(get_2q_biased_erasure_mechanism(p_e))
        else:
            mechanism_list.append(get_2q_erasure_mechanism(p_e))
    return GateErrorModel(mechanism_list)

