import numpy as np
import stim
from dataclasses import dataclass, field
from typing import Optional
from FlexibleQECSim.error_model import *
from FlexibleQECSim.qecc_circ_builder import *
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
class RotatedSurfaceCodeWithNativeParityChecksMemoryExperimentBuilder(QECCircuitBuilder):
    """
    This class is used to build the circuits for the rotated surface code with native parity checks memory experiment.
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

    def get_normal_qubit_count(self):
        return 2*(self.distance+1)**2
    
    def get_num_normal_measurements(self):
        num_data_q = self.distance**2
        num_meas_q = num_data_q-1
        num_existing_measurement = num_meas_q*(self.rounds+1)+num_data_q
        return num_existing_measurement
    
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
                    if self.native_cx:
                        append_cnot(
                            circuit=circuit,
                            qubits=dict['CX'], 
                            noisy=noisy,
                            noise_model=self.after_cnot_error_model,
                            mode=mode,
                        )
                    else:
                        hadamard_target_qubits = dict['CX'][1::2]
                        append_H(circuit = circuit, qubits = hadamard_target_qubits, noisy=noisy, noise_model=self.after_h_error_model, mode=mode)
                        append_cz(circuit = circuit, qubits = dict['CX'], noisy=noisy, noise_model=self.after_cz_error_model, mode=mode)
                        append_H(circuit = circuit, qubits = hadamard_target_qubits, noisy=noisy, noise_model=self.after_h_error_model, mode=mode)
                if dict['CZ'] != []:
                    if self.native_cz:
                        append_cz(
                            circuit=circuit,
                            qubits=dict['CZ'], 
                            noisy=noisy,
                            noise_model=self.after_cz_error_model,
                            mode=mode,
                            native_cz=self.native_cz
                        )
                    else:
                        hadamard_target_qubits = dict['CZ'][1::2]
                        append_H(circuit = circuit, qubits = hadamard_target_qubits, noisy=noisy, noise_model=self.after_h_error_model, mode=mode)
                        append_cnot(circuit = circuit, qubits = dict['CZ'], noisy=noisy, noise_model=self.after_cz_error_model, mode=mode)
                        append_H(circuit = circuit, qubits = hadamard_target_qubits, noisy=noisy, noise_model=self.after_h_error_model, mode=mode)
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
                append_measure(circuit=circuit, qubits=self.helper.data_qubits, basis="ZX"[self.is_memory_x], noisy=self.SPAM, measurement_error=self.measurement_error, mode=mode)
            else:
                # Whether measuring in Z or X basis has to do with whether the qubit was reset in the circuit head in Z or X basis
                for q in self.helper.data_qubits:
                    measure_in_Z_when_memory_x = self.helper.data_qubit_to_MX_or_MZ_in_XZZX[q]
                    measure_in_Z = measure_in_Z_when_memory_x if self.is_memory_x else not measure_in_Z_when_memory_x
                    append_measure(circuit=circuit, qubits=[q], basis="ZX"[measure_in_Z], noisy=self.SPAM, measurement_error=self.measurement_error, mode=mode)

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


