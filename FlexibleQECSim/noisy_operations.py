import stim
from typing import List, string
from FlexibleQECSim.error_model import GateErrorModel


def append_before_round_error(circuit: stim.Circuit,
                              qubits: List[int],
                              noisy: bool,
                              noise_model: GateErrorModel,
                              mode: string):
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
             mode: string):
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
                mode: string,
                native_cx: bool):
    if native_cx:
        circuit.append('CNOT', qubits)
        if noisy and not noise_model.trivial:
            list_of_args = noise_model.get_instruction(qubits=qubits,
                                                                    mode=mode,
                                                    )
            for args in list_of_args:
                circuit.append(*args)
    else:
        # control_qubits = qubits[0::2]
        target_qubits = qubits[1::2]
        # control_target_pairs = list(zip(control_qubits,target_qubits ))
        append_H(target_qubits, noisy=noisy)
        append_cz(qubits, noisy=noisy)
        append_H(target_qubits, noisy=noisy)

def append_cz(circuit: stim.Circuit, 
             qubits: List[int],
             noisy: bool, 
             noise_model: GateErrorModel,
             mode: string,
             native_cz: bool):
    if native_cz:
        circuit.append('CZ', qubits)
        if noisy and not noise_model.trivial:
            list_of_args = noise_model.get_instruction(qubits=qubits,
                                                        mode=mode,
                                                        )
            for args in list_of_args:
                circuit.append(*args)
    else:
        # control_qubits = qubits[0::2]
        target_qubits = qubits[1::2]
        # control_target_pairs = list(zip(control_qubits,target_qubits ))
        append_H(target_qubits, noisy=noisy)
        append_cnot(qubits, noisy=noisy)
        append_H(target_qubits, noisy=noisy)


def append_reset(circuit: stim.Circuit, 
                 qubits: List[int],
                 noisy: bool, 
                 basis: str,
                 noise_model: GateErrorModel,
                 mode: string):
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
                   measurement_error: float):
    if noisy:
        circuit.append("M" + basis, qubits, measurement_error)
    else:
        circuit.append("M" + basis, qubits, 0)
