import stim
from typing import List, Union
from FlexibleQECSim.error_model import GateErrorModel


def append_before_round_error(circuit: stim.Circuit,
                              qubits: List[int],
                              noisy: bool,
                              noise_model: GateErrorModel,
                              mode: str):
    circuit.append("TICK")
    if noisy and not noise_model.trivial:
        list_of_args = noise_model.get_instruction(qubits=[qubits],
                                                   mode=mode,
                                                   )
        for args in list_of_args:
            circuit.append(*args)


def append_H(circuit: stim.Circuit, 
             qubits: List[int], 
             noisy: bool, 
             noise_model: GateErrorModel,
             mode: str):
    circuit.append('H', qubits)
    if noisy and not noise_model.trivial:
        list_of_args = noise_model.get_instruction(qubits=[qubits],
                                                    mode=mode,
                                                    )
        for args in list_of_args:
            circuit.append(*args)


def append_cnot(circuit: stim.Circuit, 
                qubits: List[int],
                noisy: bool,
                noise_model: GateErrorModel,
                mode: str):
    circuit.append('CNOT', qubits)
    if noisy and not noise_model.trivial:
        list_of_args = noise_model.get_instruction(qubits=qubits,
                                                                mode=mode,
                                                )
        for args in list_of_args:
            circuit.append(*args)


def append_cz(circuit: stim.Circuit, 
             qubits: List[int],
             noisy: bool, 
             noise_model: GateErrorModel,
             mode: str):
    circuit.append('CZ', qubits)
    if noisy and not noise_model.trivial:
        list_of_args = noise_model.get_instruction(qubits=qubits,
                                                    mode=mode,
                                                    )
        for args in list_of_args:
            circuit.append(*args)


def append_reset(circuit: stim.Circuit, 
                 qubits: List[int],
                 noisy: bool, 
                 basis: str,
                 noise_model: GateErrorModel,
                 mode: str):
    assert basis == "X" or basis == "Z", "basis must be X or Z"
    circuit.append("R" + basis, qubits)
    if noisy and not noise_model.trivial:
        list_of_args = noise_model.get_instruction(qubits=qubits,
                                                    mode=mode,
                                                    )
        for args in list_of_args:
            circuit.append(*args)

def append_measure(circuit: stim.Circuit, 
                   qubits: List[int],
                   noisy: bool, 
                   basis: str,
                   measurement_error: float,
                   mode:str = None):
    if noisy:
        circuit.append("M" + basis, qubits, measurement_error)
    else:
        circuit.append("M" + basis, qubits, 0)
