import numpy as np
from typing import List, Dict, Any, Callable
import stim
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
from FlexibleQECSim.qecc_circ_builder import *
import time
from FlexibleQECSim.noisy_operations import (
    append_before_round_error,
    append_H,
    append_cnot,
    append_cz,
    append_reset,
    append_measure
)
from FlexibleQECSim.rotated_surface_helper import RotatedSurfaceCodeHelper



@dataclass
class RotatedSurfaceCodeMemoryExperimentBuilder(QECCircuitBuilder):
    """
    What is this class? 
    An instance of RotatedSurfaceCodeMemoryExperimentBuilder is generated, sent to an HTC condor node, and outputs a JSON file describing the job_id (node_id), circuit_id and the number of errors it sampled.
    One circuit corresponds to different jobs.

    An barely-initialized instance of this class should 
    1. asemble its rotated_surface_code_circuit_helper
    2. assemble its erasure circuit, sample,
    3. decode using Z and new_circ method
    4. write results to JSON.
    """

    # These attributes are optional and I tend not to change them
    interaction_order: str = 'z'  # Is 'clever' really better than z?
    native_cz: bool = True
    native_cx: bool = False
    XZZX: bool = True
    is_memory_x: bool = True
    prefer_hadamard_on_control_when_only_native_cnot_in_XZZX: bool = False
    SPAM: bool = False

    # These attributes will be generated when sampling or decoding.
    helper: Optional[RotatedSurfaceCodeHelper] = field(
        init=False, repr=False)
    erasure_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    normal_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    posterior_circuit: Optional[stim.Circuit] = field(init=False, repr=False)
    deterministic_circuit: Optional[stim.Circuit] = field(
        init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        assert (any([self.native_cz, self.native_cx]))
        # At this point an instance of this class will have all the information needed to sample and decode a particular circuit on a Node.

    def generate_helper(self):
        self.helper = RotatedSurfaceCodeHelper(rounds=self.rounds, distance=self.distance, XZZX=self.XZZX, native_cx=self.native_cx,
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
                attr_value.set_next_ancilla_qubit_index_in_list(
                    self.next_ancilla_qubit_index_in_list)
        self.erasure_circuit = stim.Circuit()

        self.gen_circuit(self.erasure_circuit, mode='erasure')
        self.erasure_circuit.append("MZ",
                                    np.arange(
                                        2*(self.distance+1)**2, self.next_ancilla_qubit_index_in_list[0], dtype=int)
                                    )  # Measure the virtual erasure ancilla qubits

    def gen_normal_circuit(self):
        # The normal circuit is only used to generate the static DEM which is then modified by the "naive" or 'Z' decoding method.
        self.normal_circuit = stim.Circuit()
        self.gen_circuit(self.normal_circuit, mode='normal')

    def gen_dummy_circuit(self):
        # The normal circuit is only used to generate the static DEM which is then modified by the "naive" or 'Z' decoding method.
        self.dummy_circuit = stim.Circuit()
        self.gen_circuit(self.dummy_circuit, mode='dummy')

    def gen_posterior_circuit(self, single_measurement_sample):
        assert len(
            single_measurement_sample) == self.erasure_circuit.num_measurements

        # Share a new erasure_measurement_index and the single_measurement_sample to the error models
        num_data_q = self.distance**2
        num_meas_q = num_data_q-1
        num_existing_measurement = num_meas_q*(self.rounds+1)+num_data_q
        self.erasure_measurement_index_in_list = [num_existing_measurement]
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

    def gen_circuit(self, circuit, mode):
        # function that builds the 1 round of error correction
        def append_cycle_actions(noisy: bool):
            append_before_round_error(
                circuit=circuit,
                qubits=self.helper.data_qubits, 
                noisy=noisy,
                noise_model=self.before_round_error_model,
                mode=mode
            )
            append_H(
                circuit=circuit,
                qubits=self.helper.meas_q_with_before_and_after_round_H, 
                noisy=noisy,
                noise_model=self.after_h_error_model,
                mode=mode
            )
            # a dict is like {'CX':[control0,target0,control1,target1,....],'CZ':[control0,target0,control1,target1,....]}
            for dict in self.helper.two_q_gate_targets:
                if dict['CX'] != []:
                    append_cnot(
                        circuit=circuit,
                        qubits=dict['CX'], 
                        noisy=noisy,
                        noise_model=self.after_cnot_error_model,
                        mode=mode,
                        native_cx=self.native_cx
                    )
                if dict['CZ'] != []:
                    append_cz(
                        circuit=circuit,
                        qubits=dict['CZ'], 
                        noisy=noisy,
                        noise_model=self.after_cz_error_model,
                        mode=mode,
                        native_cz=self.native_cz
                    )
            append_H(
                circuit=circuit,
                qubits=self.helper.meas_q_with_before_and_after_round_H, 
                noisy=noisy,
                noise_model=self.after_h_error_model,
                mode=mode
            )
            append_measure(
                circuit=circuit,
                qubits=self.helper.measurement_qubits, 
                noisy=noisy,
                basis="Z",
                measurement_error=self.measurement_error
            )
            append_reset(
                circuit=circuit,
                qubits=self.helper.measurement_qubits, 
                noisy=noisy,
                basis="Z",
                noise_model=self.after_reset_error_model,
                mode=mode
            )

        def build_circ():
            ###################################################
            # Build the circuit head and first noiseless round
            ###################################################
            for q, coord in self.helper.q2p.items():
                circuit.append("QUBIT_COORDS", [q], [coord.real, coord.imag])
            if not self.XZZX:
                append_reset(
                    circuit=circuit,
                    qubits=self.helper.data_qubits,
                    basis="ZX"[self.is_memory_x],
                    noisy=self.SPAM,
                    noise_model=self.after_reset_error_model,
                    mode=mode
                )
            else:
                X_reset_data_q = [q for q in self.helper.data_qubits if self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]]
                Z_reset_data_q = [q for q in self.helper.data_qubits if not self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]]
                if self.is_memory_x:
                    append_reset(circuit=circuit, qubits=X_reset_data_q, basis="X", noisy=self.SPAM, noise_model=self.after_reset_error_model, mode=mode)
                    append_reset(circuit=circuit, qubits=Z_reset_data_q, basis="Z", noisy=self.SPAM, noise_model=self.after_reset_error_model, mode=mode)
                else:
                    append_reset(circuit=circuit, qubits=X_reset_data_q, basis="Z", noisy=self.SPAM, noise_model=self.after_reset_error_model, mode=mode)
                    append_reset(circuit=circuit, qubits=Z_reset_data_q, basis="X", noisy=self.SPAM, noise_model=self.after_reset_error_model, mode=mode)

            append_reset(circuit=circuit, qubits=self.helper.measurement_qubits, basis="Z", noisy=self.SPAM, noise_model=self.after_reset_error_model, mode=mode)
            if self.SPAM == False:  # Shurti Puri's biased erasure paper has a noiseless round to "initialize the qubit, but Kubica's paper doesn't"
                append_cycle_actions(noisy=False)
            else:
                append_cycle_actions(noisy=True)
            # In the first round, the detectors have the same value of the measurements
            for measure in self.helper.chosen_basis_measure_coords:
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(-len(self.helper.measurement_qubits) +
                                     self.helper.measure_coord_to_order[measure])],
                    [measure.real, measure.imag, 0]
                )
            ###################################################
            # Build the repeated noisy body of the circuit, including the detectors comparing to previous cycles.
            ###################################################
            for _ in range(self.rounds-self.SPAM):  # The rest noisy rounds
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
                append_measure(circuit=circuit, qubits=self.helper.data_qubits, basis="ZX"[self.is_memory_x], noisy=self.SPAM, noise_model=self.after_measurement_error_model, mode=mode)
            else:
                # Whether measuring in Z or X basis has to do with whether the qubit was reset in the circuit head in Z or X basis
                for q in self.helper.data_qubits:
                    measure_in_Z_when_memory_x = self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]
                    measure_in_Z = measure_in_Z_when_memory_x if self.is_memory_x else not measure_in_Z_when_memory_x
                    append_measure(circuit=circuit, qubits=[q], basis="ZX"[measure_in_Z], noisy=self.SPAM, noise_model=self.after_measurement_error_model, mode=mode)

            # In CSS surface code, only physical Z error can cause logical Z error,
            #   and physical Z error are only picked up by X stabilizers,
            #   which are in chosen_basis_measure_coords
            # For XZZX code, there's no X or Z observable, but horizontal and vertical observable, but it works the same
            for measure in self.helper.chosen_basis_measure_coords:
                detectors = []
                for delta in self.helper.z_order:
                    data = measure + delta
                    if data in self.helper.p2q:
                        detectors.append(
                            len(self.helper.data_qubits) - self.helper.data_coord_to_order[data])
                detectors.append(len(self.helper.data_qubits) + len(
                    self.helper.measurement_qubits) - self.helper.measure_coord_to_order[measure])
                detectors.sort()
                list_of_records = []
                for d in detectors:
                    list_of_records.append(stim.target_rec(-d))
                circuit.append("DETECTOR", list_of_records, [
                               measure.real, measure.imag, 1])

            # Logical observable.
            obs_inc = [len(self.helper.data_qubits) - self.helper.data_coord_to_order[q]
                       for q in self.helper.chosen_basis_observable_coords]
            obs_inc.sort()
            list_of_records = []
            for obs in obs_inc:
                list_of_records.append(stim.target_rec(-obs))
            circuit.append("OBSERVABLE_INCLUDE", list_of_records, 0)

        build_circ()

    def decode_by_generate_new_circ(self, single_detector_sample, curve, single_measurement_sample):
        assert curve in ['S', 'L']
        conditional_circ = self.gen_posterior_circuit(
            single_measurement_sample)
        dem = conditional_circ.detector_error_model(
            approximate_disjoint_errors=True, decompose_errors=True)
        m = DEM_to_Matching(dem, curve=curve)
        predicted_observable = m.decode(single_detector_sample)[0]
        return predicted_observable
