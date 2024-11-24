import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from itertools import chain
from FlexibleQECSim.qecc_circ_builder import CircuitStructureHelper
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
class RotatedSurfaceCodeHelper(CircuitStructureHelper):
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
        data_coord_to_order: Dict[complex, int] = {
            q2p[q]: i for i, q in enumerate(data_qubits)}
        measure_coord_to_order: Dict[complex, int] = {
            q2p[q]: i for i, q in enumerate(measurement_qubits)}

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
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[data], p2q[measure]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
        # 2
        elif native_cx and not native_cz and XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in [0, 3]:  # X
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
            for k in [1, 2]:  # Use CZ here, the native CNOT will have its target qubit sandwiched by H in the builder
                if not prefer_hadamard_on_control_when_only_native_cnot_in_XZZX:
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                        [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                        [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
                else:
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                        [p2q[data], p2q[measure]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                    two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                        [p2q[data], p2q[measure]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
        # 3
        elif not native_cx and native_cz and not XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in range(4):
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
        # 4
        elif not native_cx and native_cz and XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in [0, 3]:  # Use CX here, the native CX will have its target qubit sandwiched by H in the builder
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
            for k in [1, 2]:
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))

        # 5
        elif native_cx and native_cz and not XZZX:
            meas_q_with_before_and_after_round_H = x_measurement_qubits
            for k in range(4):  # We can use CX and CZ, not implemented here
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[data], p2q[measure]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))

        # 6
        elif native_cx and native_cz and XZZX:
            meas_q_with_before_and_after_round_H = measurement_qubits
            for k in [0, 3]:
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CX'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))
            for k in [1, 2]:
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in x_measure_coords if (data := measure + x_order[k]) in p2q))
                two_q_gate_targets[k]['CZ'].extend(chain.from_iterable(
                    [p2q[measure], p2q[data]] for measure in z_measure_coords if (data := measure + z_order[k]) in p2q))

        self.data_qubit_to_MX_or_MZ_in_XZZX = {}
        for q in data_qubits:
            self.data_qubit_to_MX_or_MZ_in_XZZX[q] = assign_MX_or_MZ_to_data_qubit_in_XZZX(
                coord=q2p[q])

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
