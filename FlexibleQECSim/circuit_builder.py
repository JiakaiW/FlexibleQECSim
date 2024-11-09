import numpy as np
from typing import List, Dict, Any, Callable
import stim
import math
import pymatching
import re
import copy
import networkx as nx
import re
import zipfile
from itertools import chain
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Union, Optional
import os
import json
import sys
import zipfile
import pickle
from FlexibleQECSim.error_model import *
import time


def assign_MX_or_MZ_to_data_qubit_in_XZZX(coord):
    ###################################################
    # Contrary to CSS surface code where the data qubits are all initialized in |0> or |->,
    #  XZZX code initialized in X and Z basis in a checkerboard pattern
    #  I determine whether to initialize and measure a data qubit according to https://quantumcomputing.stackexchange.com/questions/28725/whats-the-logical-x-and-z-operator-in-xzzx-surface-code
    ###################################################
    real_part = int(coord.real) % 4
    imag_part = int(coord.imag) % 4

    if (real_part == 1 and imag_part == 3) or (real_part == 3 and imag_part == 1):
        return True
    else:
        return False

@dataclass
class rotated_surface_code_circuit_helper:
    """
    This function generates some helpful information like qubit location and indices
    """
    rounds: int
    distance: int
    XZZX: bool
    native_cx: bool
    native_cz: bool
    interaction_order: str  # 'x','z','clever'
    is_memory_x: bool = True
    prefer_hadamard_on_control_when_only_native_cnot_in_XZZX: bool = False

    def __post_init__(self):

        d = self.distance
        distance = self.distance
        rounds = self.rounds
        XZZX = self.XZZX
        native_cx = self.native_cx
        native_cz = self.native_cz
        interaction_order = self.interaction_order
        is_memory_x = self.is_memory_x
        is_memory_x = self.is_memory_x
        prefer_hadamard_on_control_when_only_native_cnot_in_XZZX = self.prefer_hadamard_on_control_when_only_native_cnot_in_XZZX
        # Place data qubits.
        data_coords = []
        x_observable_coords = []
        z_observable_coords = []
        for x in np.arange(0.5, d, 1):
            for y in np.arange(0.5, d, 1):
                q = x * 2 + y * 2j
                data_coords.append(q)
                if y == 0.5:
                    z_observable_coords.append(q)
                if x == 0.5:
                    x_observable_coords.append(q)

        # Place measurement qubits.
        x_measure_coords = []
        z_measure_coords = []
        for x in range(d + 1):
            for y in range(d + 1):
                q = x * 2 + y * 2j
                on_boundary_1 = x == 0 or x == d
                on_boundary_2 = y == 0 or y == d
                parity = x % 2 != y % 2
                if on_boundary_1 and parity:
                    continue
                if on_boundary_2 and not parity:
                    continue
                if parity:
                    x_measure_coords.append(q)
                else:
                    z_measure_coords.append(q)

        # Define interaction orders so that hook errors run against the error grain instead of with it.
        z_order = [
            1 + 1j,  # br
            1 - 1j,  # tr
            -1 + 1j,  # bl
            -1 - 1j,  # tl
        ]
        x_order = [
            1 + 1j,  # br
            -1 + 1j,  # bl
            1 - 1j,  # tr
            -1 - 1j,  # tl
        ]

        if interaction_order == 'z':
            x_order = z_order
        elif interaction_order == 'x':
            z_order = x_order
        elif interaction_order == 'clever':
            pass

        def coord_to_index(q: complex) -> int:
            q = q - (0 + q.real % 2 * 1j)
            return int(q.real + q.imag * (d + 0.5))

        if rounds < 1:
            raise ValueError("Need rounds >= 1.")
        if distance < 2:
            raise ValueError("Need a distance >= 2.")

        chosen_basis_observable_coords = x_observable_coords if is_memory_x else z_observable_coords
        chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords

        # Index the measurement qubits and data qubits.
        p2q: Dict[complex, int] = {}
        for q in data_coords:
            p2q[q] = coord_to_index(q)
        for q in x_measure_coords:
            p2q[q] = coord_to_index(q)
        for q in z_measure_coords:
            p2q[q] = coord_to_index(q)

        # Reverse index.
        q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}

        # Make target lists for various types of qubits.
        data_qubits: List[int] = [p2q[q] for q in data_coords]
        measurement_qubits: List[int] = [p2q[q] for q in x_measure_coords]
        x_measurement_qubits: List[int] = [p2q[q] for q in x_measure_coords]
        measurement_qubits += [p2q[q] for q in z_measure_coords]
        all_qubits: List[int] = data_qubits + measurement_qubits
        all_qubits.sort()
        data_qubits.sort()
        measurement_qubits.sort()
        x_measurement_qubits.sort()

        # Reverse index the measurement order used for defining detectors.
        data_coord_to_order: Dict[complex, int] = {q2p[q]: i for i, q in enumerate(data_qubits)}
        measure_coord_to_order: Dict[complex, int] = {q2p[q]: i for i, q in enumerate(measurement_qubits)}

        # List CNOT or CZ gate targets using given interaction orders.
        # [{'CX':[control0,target0,control1,target1,....],'CZ':[control0,target0,control1,target1,....]},
        # {'CX':[control0,target0,control1,target1,....],'CZ':[control0,target0,control1,target1,....]},
        # {'CX':[control0,target0,control1,target1,....],'CZ':[control0,target0,control1,target1,....]},
        # {'CX':[control0,target0,control1,target1,....],'CZ':[control0,target0,control1,target1,....]}]
        two_q_gate_targets = [{'CX': [], 'CZ': []},
                              {'CX': [], 'CZ': []},
                              {'CX': [], 'CZ': []},
                              {'CX': [], 'CZ': []}, ]
        meas_q_with_before_and_after_round_H = None
        # List which measurement qubits need to be applied a H before and after each round
        # 1
        if native_cx and not native_cz and not XZZX:  # Original plan in stim.generate
            meas_q_with_before_and_after_round_H = x_measurement_qubits
            for k in range(4):
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[data], p2q[measure]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
        # 2
        elif native_cx and not native_cz and XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in [0, 3]:  # X
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
            for k in [1, 2]:  # Use CZ here, the native CNOT will have its target qubit sandwiched by H in the builder
                if not prefer_hadamard_on_control_when_only_native_cnot_in_XZZX:
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
                else:
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[data], p2q[measure]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[data], p2q[measure]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
        # 3
        elif not native_cx and native_cz and not XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in range(4):
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
        # 4
        elif not native_cx and native_cz and XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in [0, 3]:  # Use CX here, the native CX will have its target qubit sandwiched by H in the builder
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
            for k in [1, 2]:
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))

        # 5
        elif native_cx and native_cz and not XZZX:
            meas_q_with_before_and_after_round_H = x_measurement_qubits
            for k in range(4):  # We can use CX and CZ, not implemented here
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[data], p2q[measure]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))

        # 6
        elif native_cx and native_cz and XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in [0, 3]:
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
            for k in [1, 2]:
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable([p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))

        self.data_qubit_to_MX_or_MZ_in_XZZX = {}
        for q in data_qubits:
            self.data_qubit_to_MX_or_MZ_in_XZZX[q] = assign_MX_or_MZ_to_data_qubit_in_XZZX(coord=q2p[q])

        self.q2p = q2p
        self.p2q = p2q
        self.data_qubits = data_qubits
        self.is_memory_x = is_memory_x
        self.meas_q_with_before_and_after_round_H = meas_q_with_before_and_after_round_H
        self.x_measurement_qubits = x_measurement_qubits
        self.measurement_qubits = measurement_qubits
        self.chosen_basis_measure_coords = chosen_basis_measure_coords
        self.chosen_basis_observable_coords = chosen_basis_observable_coords
        self.measure_coord_to_order = measure_coord_to_order
        self.data_coord_to_order = data_coord_to_order
        self.z_order = z_order  # this can be any order because when it's used by a circ_builder, it's only used to get all data qubits in the stabilizer
        self.two_q_gate_targets = two_q_gate_targets


def DEM_to_Matching(model: stim.DetectorErrorModel,
                             single_measurement_sample: np.array = None,
                             detectors_to_list_of_meas = None,
                             erasure_handling = None,
                             curve = 'L'
                             ) -> pymatching.Matching:
    """
    This method will be used by the builder/decoder (they are one class now. The class instace is sent to every condor node)
    Because there are unconnected nodes in the matching graph if I do one round of noiseless correction before final logical measurement, I have to construct the graph manually
    Modified from Craig Gidney's code: https://gist.github.com/Strilanc/a4a5f2f9410f84212f6b2c26d9e46e24/
    and https://github.com/Strilanc/honeycomb-boundaries/blob/main/src/hcb/tools/analysis/decoding.py#L260
    """
    det_offset = 0

    def _iter_model(m: stim.DetectorErrorModel,
                    reps: int,
                    handle_error: Callable[[float, List[int], List[int]], None]):
        nonlocal det_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _iter_model(instruction.body_copy(), instruction.repeat_count, handle_error)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets: List[int] = []
                        frames: List[int] = []
                        t: stim.DemTarget
                        p = instruction.args_copy()[0]
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames.append(t.val)
                            elif t.is_separator():
                                handle_error(p, dets, frames)
                                frames = []
                                dets = []
                        handle_error(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                    elif instruction.type == "detector":
                        pass
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

    g = nx.Graph()
    num_detectors = model.num_detectors
    for k in range(num_detectors):
        g.add_node(k)
    g.add_node(num_detectors, is_boundary=True)
    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 0:
            return
        if len(dets) == 1:
            dets.append(num_detectors) # boundary edge, the other node is a "boundary" node
        if len(dets) > 2:
            print(f'len dets > 2: {dets}')
            return
        

        # dets_str = str(sorted(dets))

        if erasure_handling == None: 
            #Used when not changing weights or the dem is already re-constructed
            if g.has_edge(*dets):
                edge_data = g.get_edge_data(*dets)
                old_p = edge_data["error_probability"]
                old_frame_changes = edge_data["qubit_id"]
                # If frame changes differ, the code has distance 2; just keep whichever was first.
                if set(old_frame_changes) != set(frame_changes):
                    frame_changes = old_frame_changes
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)

        else:
            Exception('unimplemented weight assignment method')

        if p > 1-1e-10:
            p =  1-1e-10
        elif p < 1e-10:
            p = 1e-10
        if curve == 'S':
            weight=math.log((1 - p) / p)
        elif curve == 'L':
            weight = -math.log(p)
        if erasure_handling == None:
            g.add_edge(*dets, weight=weight, qubit_id=frame_changes, error_probability=p)
        # else:
        #     g.add_edge(*dets, weight=weight, qubit_id=frame_changes, error_probability=p, erased = erasure)
        ## end of handle_error()

    _iter_model(model, 1, handle_error)
    return pymatching.Matching(g)


@dataclass
class easure_circ_builder:
    """
    What is this class? 
    An instance of easure_circ_builder is generated, sent to an HTC condor node, and outputs a JSON file describing the job_id (node_id), circuit_id and the number of errors it sampled.
    One circuit corresponds to different jobs.

    An barely-initialized instance of this class should 
    1. asemble its rotated_surface_code_circuit_helper
    2. assemble its erasure circuit, sample,
    3. decode using Z and new_circ method
    4. write results to JSON.
    """
    rounds: int
    distance: int

    before_round_error_model: GateErrorModel  = GateErrorModel([])
    after_h_error_model: GateErrorModel  = GateErrorModel([])
    after_cnot_error_model: GateErrorModel  = GateErrorModel([])
    after_cz_error_model: GateErrorModel  = GateErrorModel([])
    measurement_error: float  = 0
    after_reset_error_model: GateErrorModel  = GateErrorModel([])

    # These attributes are optional and I tend not to change them
    interaction_order: str = 'z' # Is 'clever' really better than z?
    native_cz: bool = True
    native_cx: bool = False
    XZZX: bool = True
    is_memory_x: bool = True
    prefer_hadamard_on_control_when_only_native_cnot_in_XZZX: bool = False
    SPAM: bool = False

    # These attributes will be generated when sampling or decoding.
    helper: Optional[rotated_surface_code_circuit_helper] = field(init=False, repr=False)
    erasure_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    normal_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    posterior_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    deterministic_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    def __post_init__(self):
        assert(any([self.native_cz,self.native_cx]))
        # At this point an instance of this class will have all the information needed to sample and decode a particular circuit on a Node.

    def generate_helper(self):
        self.helper = rotated_surface_code_circuit_helper(rounds=self.rounds, distance=self.distance, XZZX=self.XZZX, native_cx=self.native_cx,
                                                          native_cz=self.native_cz,
                                                          interaction_order=self.interaction_order,
                                                          is_memory_x=self.is_memory_x,
                                                          prefer_hadamard_on_control_when_only_native_cnot_in_XZZX=self.prefer_hadamard_on_control_when_only_native_cnot_in_XZZX)

        # self.gen_erasure_conversion_circuit()
        # self.gen_normal_circuit()


    def gen_erasure_conversion_circuit(self):
        # erasure_circuit is used to sample measurement samples which we do decoding on
        self.next_ancilla_qubit_index_in_list = [2*(self.distance+1)**2]
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, GateErrorModel):
                attr_value.set_next_ancilla_qubit_index_in_list(self.next_ancilla_qubit_index_in_list)
        self.erasure_circuit = stim.Circuit()
    
        self.gen_circuit(self.erasure_circuit, mode = 'erasure')
        self.erasure_circuit.append("MZ", 
                                    np.arange(2*(self.distance+1)**2, self.next_ancilla_qubit_index_in_list[0], dtype=int)
                                    )  # Measure the virtual erasure ancilla qubits


    def gen_normal_circuit(self):
        # The normal circuit is only used to generate the static DEM which is then modified by the "naive" or 'Z' decoding method.
        self.normal_circuit = stim.Circuit()
        self.gen_circuit(self.normal_circuit, mode = 'normal')

    def gen_dummy_circuit(self):
        # The normal circuit is only used to generate the static DEM which is then modified by the "naive" or 'Z' decoding method.
        self.dummy_circuit = stim.Circuit()
        self.gen_circuit(self.dummy_circuit, mode = 'dummy')


    def gen_posterior_circuit(self,single_measurement_sample):
        assert len(single_measurement_sample) == self.erasure_circuit.num_measurements 

        # Share a new erasure_measurement_index and the single_measurement_sample to the error models
        num_data_q = self.distance**2
        num_meas_q = num_data_q-1
        num_existing_measurement = num_meas_q*(self.rounds+1)+num_data_q
        self.erasure_measurement_index_in_list = [num_existing_measurement]
        self.single_shot_measurement_sample_being_decoded = single_measurement_sample
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, GateErrorModel):
                attr_value.set_erasure_measurement_index_in_list(self.erasure_measurement_index_in_list)
                attr_value.set_single_measurement_sample(self.single_shot_measurement_sample_being_decoded)

        self.posterior_circuit = stim.Circuit()
        self.gen_circuit(self.posterior_circuit, mode = 'posterior')
        assert self.erasure_measurement_index_in_list[0] == self.erasure_circuit.num_measurements
        return self.posterior_circuit
    

    def gen_circuit(self, circuit, mode):
        def append_before_round_error(data_qubits: List[int],
                                      noisy: bool):
            circuit.append("TICK")
            if noisy and not self.before_round_error_model.trivial:
                list_of_args = self.before_round_error_model.get_instruction(qubits = [data_qubits],
                                                                            mode=mode,
                                                                            )
                for args in list_of_args:
                    circuit.append(*args)

        def append_H(targets: List[int],
                     noisy: bool):
            circuit.append('H', targets)
            if noisy and not self.after_h_error_model.trivial:
                list_of_args = self.after_h_error_model.get_instruction(qubits = [targets],
                                                                            mode=mode,
                                                                            )
                for args in list_of_args:
                    circuit.append(*args)

        def append_cnot(qubits: List[int],
                        noisy: bool):
            if self.native_cx:
                circuit.append('CNOT', qubits)
                if noisy and not self.after_cnot_error_model.trivial:
                    list_of_args = self.after_cnot_error_model.get_instruction(qubits = qubits,
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

        def append_cz(qubits: List[int],
                      noisy: bool):
            if self.native_cz:
                circuit.append('CZ', qubits)
                if noisy and not self.after_cz_error_model.trivial:
                    list_of_args = self.after_cz_error_model.get_instruction(qubits = qubits,
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

        def append_reset(targets: List[int], basis: str, noisy: bool):
            assert basis == "X" or basis == "Z", "basis must be X or Z"
            circuit.append("R" + basis, targets)
            if noisy and not self.after_reset_error_model.trivial:
                list_of_args =  self.after_reset_error_model.get_instruction(qubits = targets,
                                                                            mode=mode,
                                                                            )
                for args in list_of_args:
                    circuit.append(*args)
        def append_measure(targets: List[int], basis: str, noisy: bool):
            if noisy:
                circuit.append("M" + basis, targets, self.measurement_error)
            else:
                circuit.append("M" + basis, targets, 0)

        # function that builds the 1 round of error correction
        def append_cycle_actions(noisy: bool):
            append_before_round_error(self.helper.data_qubits, noisy)
            append_H(self.helper.meas_q_with_before_and_after_round_H, noisy)
            for dict in self.helper.two_q_gate_targets:  # a dict is like {'CX':[control0,target0,control1,target1,....],'CZ':[control0,target0,control1,target1,....]}
                if dict['CX'] != []:
                    append_cnot(dict['CX'], noisy)
                if dict['CZ'] != []:
                    append_cz(dict['CZ'], noisy)
            append_H(self.helper.meas_q_with_before_and_after_round_H, noisy)
            append_measure(self.helper.measurement_qubits, "Z", noisy)
            append_reset(self.helper.measurement_qubits, "Z", noisy)
            

        def build_circ():
            ###################################################
            # Build the circuit head and first noiseless round
            ###################################################
            for q, coord in self.helper.q2p.items():
                circuit.append("QUBIT_COORDS", [q], [coord.real, coord.imag])
            if not self.XZZX:
                append_reset(self.helper.data_qubits, "ZX"[self.is_memory_x], noisy=self.SPAM)
            else:
                X_reset_data_q = [q for q in self.helper.data_qubits if self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]]
                Z_reset_data_q = [q for q in self.helper.data_qubits if not self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]]
                if self.is_memory_x:
                    append_reset(X_reset_data_q, "X", noisy=self.SPAM)
                    append_reset(Z_reset_data_q, "Z", noisy=self.SPAM)
                else:
                    append_reset(X_reset_data_q, "Z", noisy=self.SPAM)
                    append_reset(Z_reset_data_q, "X", noisy=self.SPAM)

            append_reset(self.helper.measurement_qubits, "Z", noisy=self.SPAM)
            if self.SPAM == False: # Shurti Puri's biased erasure paper has a noiseless round to "initialize the qubit, but Kubica's paper doesn't"
                append_cycle_actions(noisy=False)
            else:
                append_cycle_actions(noisy=True)
            # In the first round, the detectors have the same value of the measurements
            for measure in self.helper.chosen_basis_measure_coords:
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(-len(self.helper.measurement_qubits) + self.helper.measure_coord_to_order[measure])],
                    [measure.real, measure.imag, 0]
                )
            ###################################################
            # Build the repeated noisy body of the circuit, including the detectors comparing to previous cycles.
            ###################################################
            for _ in range(self.rounds-self.SPAM): # The rest noisy rounds
                append_cycle_actions(noisy=True)
                circuit.append("SHIFT_COORDS", [], [0, 0, 1])
                m = len(self.helper.measurement_qubits)
                # The for loop below calculate the relative measurement indexes to set up the detectors
                for m_index in self.helper.measurement_qubits:
                    m_coord = self.helper.q2p[m_index]
                    k = m - self.helper.measure_coord_to_order[m_coord] - 1
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                        [m_coord.real, m_coord.imag, 0]
                    )
            ###################################################
            # In Kubica (Amazon) paper, they do a final noiseless round after d noisy round.
            # But in Shurti Puri paper, they do d noisy round and only final noiseless measurement. (What's done below.)
            ###################################################

            ###################################################
            # Build the end of the circuit, getting out of the cycle state and terminating.
            # In particular, the data measurements create detectors that have to be handled specially.
            # Also, the tail is responsible for identifying the logical observable.
            ###################################################
            if not self.XZZX:
                append_measure(self.helper.data_qubits, "ZX"[self.is_memory_x], noisy=self.SPAM)
            else:
                # Whether measuring in Z or X basis has to do with whether the qubit was reset in the circuit head in Z or X basis
                for q in self.helper.data_qubits:
                    measure_in_Z_when_memory_x = self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]
                    measure_in_Z = measure_in_Z_when_memory_x if self.is_memory_x else not measure_in_Z_when_memory_x
                    append_measure([q], "ZX"[measure_in_Z], noisy=self.SPAM)

            # In CSS surface code, only physical Z error can cause logical Z error,
            #   and physical Z error are only picked up by X stabilizers,
            #   which are in chosen_basis_measure_coords
            # For XZZX code, there's no X or Z observable, but horizontal and vertical observable, but it works the same
            for measure in self.helper.chosen_basis_measure_coords:
                detectors = []
                for delta in self.helper.z_order:
                    data = measure + delta
                    if data in self.helper.p2q:
                        detectors.append(len(self.helper.data_qubits) - self.helper.data_coord_to_order[data])
                detectors.append(len(self.helper.data_qubits) + len(self.helper.measurement_qubits) - self.helper.measure_coord_to_order[measure])
                detectors.sort()
                list_of_records = []
                for d in detectors:
                    list_of_records.append(stim.target_rec(-d))
                circuit.append("DETECTOR", list_of_records, [measure.real, measure.imag, 1])

            # Logical observable.
            obs_inc = [len(self.helper.data_qubits) - self.helper.data_coord_to_order[q] for q in self.helper.chosen_basis_observable_coords]
            obs_inc.sort()
            list_of_records = []
            for obs in obs_inc:
                list_of_records.append(stim.target_rec(-obs))
            circuit.append("OBSERVABLE_INCLUDE", list_of_records, 0)
        
        build_circ()


    def decode_by_generate_new_circ(self,single_detector_sample,curve,single_measurement_sample):
        assert curve in ['S','L']
        conditional_circ = self.gen_posterior_circuit(single_measurement_sample)
        dem = conditional_circ.detector_error_model(approximate_disjoint_errors=True,decompose_errors=True)
        m = DEM_to_Matching(dem,curve = curve)
        predicted_observable = m.decode(single_detector_sample)[0]
        return predicted_observable





    # def decode_without_changing_weights(self,single_detector_sample,curve,single_measurement_sample= None):
    #     # single_measurement_sample in the arguement is just to keep consistancy with other decoders
    #     assert curve in ['S','L']
    #     if curve == 'L':
    #         predicted_observable =self.dummy_matching_L.decode(single_detector_sample)[0]
    #     else:
    #         predicted_observable =self.dummy_matching_S.decode(single_detector_sample)[0]
    #     return predicted_observable


# def ancilla_to_detectors(erasure_circ_text: str) -> Dict[int, List[List[int]]]:
#     '''
#     Not used anymore. Now erasure decoding is handled by the Gate_error_model class

#     This function takes in the text form of an erasure stim circuit and output a dictionary that maps the erasure qubit index to the index of corresponding detectors
#     It edits the string representation of the circuit to isolate error to find out which detectors one specific error flips
#     To isolate an error, it removes all CORRELATED_ERROR and ELSE_CORRELATED_ERROR and adjusts the error rate for measurements to 0
#     '''
    
#     # Add line numbers to each line of the circuit string
#     def append_line_numbers(input_string):
#         lines = input_string.splitlines()
#         result = []
#         for i, line in enumerate(lines, start=0):
#             line = line + f" #{i}"
#             result.append(line)
#         return "\n".join(result)

#     erasure_circ_with_line_num = append_line_numbers(erasure_circ_text)

#     # Get information about the erasure error line number, error rates, and qubit-ancilla pair
#     # For example, a line like PAULI_CHANNEL_2(0, 0, 0, 0.005, 0, 0, 0, 0.005, 0, 0, 0, 0.005, 0, 0, 0) 2 26 16 27 11 28 14 29 9 30 18 31 3 32 17 33 12 34 15 35 10 36 19 37 #3
#     # will be matched to three components:
#     # 0, 0, 0, 0.005, 0, 0, 0, 0.005, 0, 0, 0, 0.005, 0, 0, 0
#     # 2 26 16 27 11 28 14 29 9 30 18 31 3 32 17 33 12 34 15 35 10 36 19 37
#     # 3
#     pattern = r'PAULI_CHANNEL_2\(([^)]+)\)\s+(.+?)\s*#(\d+)$'
#     lines = re.findall(pattern, erasure_circ_with_line_num, re.MULTILINE)
#     line_num_error_rates_data_ancilla_pairs = []
#     for floats, integers, line_index in lines:
#         float_list = [float(num) for num in floats.split(', ')]
#         integer_list = [[int(integers.split()[i]), int(integers.split()[i + 1])] for i in
#                         range(0, len(integers.split()), 2)]
#         line_num_error_rates_data_ancilla_pairs.append([int(line_index), float_list, integer_list])

#     # organize the correspondance between which data qubit have error, when, and associated with which ancilla
#     data_qubit_ancilla_line_number = []
#     erasure_line_numbers = []
#     for r in line_num_error_rates_data_ancilla_pairs:
#         erasure_line_numbers.append(r[0])
#         for pair in r[2]:
#             data_qubit_ancilla_line_number.append([pair[0], pair[1], r[0]])  # [data index, ancilla index, line number]

#     def delete_irrelavent_error_lines(match):
#         index = int(match.group(1))
#         if index in indices_to_delete:
#             return ''
#         else: # I don't know why I wrote this else. Theoratically the line index should always be in indices_to_delete.
#             return match.group(0)

#     ancilla_to_detectors = {}
#     for single_erasure in data_qubit_ancilla_line_number:
#         # isolate the error and increase probability to 1
#         line_num = single_erasure[2]
#         data_qubit_index = single_erasure[0]
#         ancilla_qubit_index = single_erasure[1]
#         error_rate = 1
#         replacement_line_pattern = r'PAULI_CHANNEL_2\([^)]+\)\s+[^#]+#{}\n'.format(line_num)
#         indices = []
#         for basis in ['X', 'Z']:
#             replacement = '{}_ERROR({}) {}\n'.format(basis, error_rate, data_qubit_index)
#             single_error_replaced = re.sub(replacement_line_pattern, replacement, erasure_circ_with_line_num)

#             # Delete all other detectable erasure errors
#             indices_to_delete = erasure_line_numbers.copy()
#             indices_to_delete.remove(line_num)
#             delete_pattern = r'PAULI_CHANNEL_2\([^)]+\)\s+[^#]+\s*#(\d+)\n'
#             erasure_error_deleted = re.sub(delete_pattern, delete_irrelavent_error_lines, single_error_replaced)
#             noisy_measurement_pattern = r'M\(\d+(\.\d+)?\)'
#             noisy_measurement_replacement = 'M(0)'
#             single_error_isolated = re.sub(noisy_measurement_pattern, noisy_measurement_replacement,
#                                            erasure_error_deleted)
#             isolated_error_circ = stim.Circuit()
#             isolated_error_circ.append_from_stim_program_text(single_error_isolated)
#             sample = isolated_error_circ.compile_detector_sampler().sample(1)[0]
#             this_shot_detectors_flagged = np.where(sample)[0].tolist()
#             indices.append(this_shot_detectors_flagged)
#         ancilla_to_detectors[ancilla_qubit_index] = indices
#     return ancilla_to_detectors
