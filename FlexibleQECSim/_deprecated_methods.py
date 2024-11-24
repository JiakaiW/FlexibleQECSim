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
