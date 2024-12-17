from dataclasses import dataclass, field
from typing import List, Optional, Callable
import stim
import math
import pymatching
import re
import copy
import networkx as nx
import re
import numpy as np
from FlexibleQECSim.error_model import GateErrorModel


class CircuitStructureHelper:
    """
    A base class for circuit structure helpers.
    """
    pass


@dataclass
class QECCircuitBuilder:
    """
    A general class for building quantum error correction circuits.
    """
    rounds: int
    distance: int

    before_round_error_model: GateErrorModel = field(default_factory=lambda: GateErrorModel([]))
    after_h_error_model: GateErrorModel = field(default_factory=lambda: GateErrorModel([]))
    after_cnot_error_model: GateErrorModel = field(default_factory=lambda: GateErrorModel([]))
    after_cz_error_model: GateErrorModel = field(default_factory=lambda: GateErrorModel([]))
    measurement_error: float = 0
    after_reset_error_model: GateErrorModel = field(default_factory=lambda: GateErrorModel([]))

    # These attributes will be generated when sampling or decoding.
    helper: Optional[CircuitStructureHelper] = field(init=False, repr=False)
    erasure_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    normal_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    posterior_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    deterministic_circuit: Optional[stim.Circuit] = field(
        init=False, repr=False)

    def __post_init__(self):
        # Initialization logic common to all QECCircuitBuilders
        pass

    def generate_helper(self):
        # Method to generate helper, to be overridden or extended by subclasses
        pass

    def gen_circuit(self, circuit, mode):
        # Method to generate a circuit, to be overridden or extended by subclasses
        pass

    def get_normal_qubit_count(self):
        pass

    def get_num_normal_measurements(self):
        pass

    def gen_erasure_conversion_circuit(self):
        # erasure_circuit is used to sample measurement samples which we do decoding on
        self.next_ancilla_qubit_index_in_list = [
            self.get_normal_qubit_count()]
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, GateErrorModel):
                attr_value.set_next_ancilla_qubit_index_in_list(
                    self.next_ancilla_qubit_index_in_list)
        self.erasure_circuit = stim.Circuit()

        self.gen_circuit(self.erasure_circuit, mode='erasure')
        self.erasure_circuit.append("MZ",
                                    np.arange(
                                        self.get_normal_qubit_count(), self.next_ancilla_qubit_index_in_list[0], dtype=int)
                                    )  # Measure the virtual erasure ancilla qubits

    def gen_normal_circuit(self):
        # The normal circuit is only used to generate the static DEM which is then modified by the "naive" or 'Z' decoding method.
        self.normal_circuit = stim.Circuit()
        self.gen_circuit(self.normal_circuit, mode='normal')

    def gen_posterior_circuit(self, single_measurement_sample):
        assert len(single_measurement_sample) == self.erasure_circuit.num_measurements

        # Share a new erasure_measurement_index and the single_measurement_sample to the error models
        self.erasure_measurement_index_in_list = [self.get_num_normal_measurements()]
        self.single_shot_measurement_sample_being_decoded = single_measurement_sample
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, GateErrorModel):
                attr_value.set_erasure_measurement_index_in_list(
                    self.erasure_measurement_index_in_list)
                attr_value.set_single_measurement_sample(
                    self.single_shot_measurement_sample_being_decoded)

        self.posterior_circuit = stim.Circuit()
        self.gen_circuit(self.posterior_circuit, mode='posterior')
        assert self.erasure_measurement_index_in_list[0] == self.erasure_circuit.num_measurements
        return self.posterior_circuit

    def decode_by_generate_new_circ(self, single_detector_sample, curve, single_measurement_sample):
        assert curve in ['S', 'L']
        conditional_circ = self.gen_posterior_circuit(
            single_measurement_sample)
        dem = conditional_circ.detector_error_model(
            approximate_disjoint_errors=True, decompose_errors=True)
        m = DEM_to_Matching(dem, curve=curve)
        predicted_observable = m.decode(single_detector_sample)[0]
        return predicted_observable

def DEM_to_Matching(model: stim.DetectorErrorModel,
                    single_measurement_sample: np.array = None,
                    detectors_to_list_of_meas=None,
                    erasure_handling=None,
                    curve='L'
                    ) -> pymatching.Matching:
    """
    This method will be used by the builder/decoder (they are one class now. The class instace is sent to every condor node)
    Because there are unconnected nodes in the matching graph if I do one round of noiseless correction before final logical measurement, I have to construct the graph manually
    Modified from Craig Gidney's code: https://gist.github.com/Strilanc/a4a5f2f9410f84212f6b2c26d9e46e24/
    and https://github.com/Strilanc/honeycomb-boundaries/blob/main/src/hcb/tools/analysis/decoding.py#L260
    """
    assert curve in ['S', 'L']
    det_offset = 0

    def _iter_model(m: stim.DetectorErrorModel,
                    reps: int,
                    handle_error: Callable[[float, List[int], List[int]], None]):
        nonlocal det_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _iter_model(instruction.body_copy(),
                                instruction.repeat_count, handle_error)
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
            # boundary edge, the other node is a "boundary" node
            dets.append(num_detectors)
        if len(dets) > 2:
            print(f'len dets > 2: {dets}')
            return

        # dets_str = str(sorted(dets))

        if erasure_handling == None:
            # Used when not changing weights or the dem is already re-constructed
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
            p = 1-1e-10
        elif p < 1e-10:
            p = 1e-10
        if curve == 'S':
            weight = math.log((1 - p) / p)
        elif curve == 'L':
            weight = -math.log(p)
        if erasure_handling == None:
            g.add_edge(*dets, weight=weight,
                       qubit_id=frame_changes, error_probability=p)
        # else:
        #     g.add_edge(*dets, weight=weight, qubit_id=frame_changes, error_probability=p, erased = erasure)
        # end of handle_error()

    _iter_model(model, 1, handle_error)
    return pymatching.Matching(g)
