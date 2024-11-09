from FlexibleQECSim.circuit_builder import *
from FlexibleQECSim.error_model import *

import time

def generate_bool_array(tot, choose):
    array = np.zeros(tot, dtype=bool)
    indices = np.random.choice(tot, choose, replace=False)
    assert len(indices) == choose
    array[indices] = True
    return array

@dataclass
class ImportanceSamplingDecodeJob:
    job_id: str
    circuit_id: str
    d: int
    p_e: float
    p_p: float
    p_z_shift: float
    p_m: float

    shots: int

    num_e_flipped: int
    num_p_flipped: int


    def sample_and_print_result(self,print_progress = False):
        if print_progress:
            from IPython.display import clear_output

        after_cz_error_model = get_2q_error_model(p_e=self.p_e,
                                                  p_p=self.p_p)
        builder = easure_circ_builder(rounds = self.d,
                                      distance= self.d,
                                      after_cz_error_model=after_cz_error_model,
                                      measurement_error=0
                                      )
        builder.generate_helper()
        builder.gen_dummy_circuit()

        non_trivial_gate_error_models = [attr_value for attr_name, attr_value in vars(builder).items() if isinstance(attr_value, GateErrorModel) and not  attr_value.trivial]
        assert len(non_trivial_gate_error_models) == 1

        tot_e = non_trivial_gate_error_models[0].name_to_mechanism['2q erasure'].dummy_generator.num_qubit_called
        tot_p = non_trivial_gate_error_models[0].name_to_mechanism['2q depo'].dummy_generator.num_qubit_called

        num_qubit_per_dice_e = non_trivial_gate_error_models[0].name_to_mechanism['2q erasure'].deterministic_generator.num_qubit_per_dice
        num_qubit_per_dice_p = non_trivial_gate_error_models[0].name_to_mechanism['2q depo'].deterministic_generator.num_qubit_per_dice

        num_dice_e = int(tot_e/num_qubit_per_dice_e)
        num_dice_p = int(tot_p/num_qubit_per_dice_p)

        builder.gen_erasure_conversion_circuit()
        erasure_circ_next_ancilla_qubit_index = builder.next_ancilla_qubit_index_in_list[0]
        converter = builder.erasure_circuit.compile_m2d_converter()

        num_shots = 0
        num_errors = 0

        for i in range(self.shots):
            e_dice_sample = generate_bool_array(num_dice_e, self.num_e_flipped)
            p_dice_sample = generate_bool_array(num_dice_p, self.num_p_flipped)

            non_trivial_gate_error_models[0].name_to_mechanism['2q erasure'].next_dice_index_in_list = [0]
            non_trivial_gate_error_models[0].name_to_mechanism['2q depo'].next_dice_index_in_list = [0]

            non_trivial_gate_error_models[0].name_to_mechanism['2q erasure'].single_dice_sample = e_dice_sample
            non_trivial_gate_error_models[0].name_to_mechanism['2q depo'].single_dice_sample = p_dice_sample

            builder.deterministic_circuit = stim.Circuit()
            builder.gen_circuit(builder.deterministic_circuit,mode='deterministic')
            builder.deterministic_circuit.append("MZ", 
                                        np.arange(2*(builder.distance+1)**2, erasure_circ_next_ancilla_qubit_index, dtype=int)
                                        )  # Measure the virtual erasure ancilla qubits
            
            sampler = builder.deterministic_circuit.compile_sampler()
            meas_samples = sampler.sample(shots=1)
            det_samples, actual_obs_chunk = converter.convert(measurements=meas_samples,
                                                                    separate_observables=True)
            
            predicted  = builder.decode_by_generate_new_circ(det_samples[0],'S',meas_samples[0])
            num_errors += actual_obs_chunk[0][0] != predicted
            num_shots += 1

        # type cast in case some of them are numpy types which are not JSON serializable
        result = {
            'job_id': str(self.job_id),
            'circuit_id': str(self.circuit_id),
            'd': int(self.d),
            'p_e':float(self.p_e),
            'p_p':float(self.p_p),
            'p_z_shift':float(self.p_z_shift),
            'p_m':float(self.p_m),
            'shots': int(num_shots),
            'num_e_flipped':int(self.num_e_flipped),
            'num_p_flipped':int(self.num_p_flipped),
            'num_shots': int(num_shots),
            'num_errors': int(num_errors),
        }

        return result

# I have another python file that loads the pickle and call the method of that class instance
# # This function can be run over condor
# def main():
#     counter = sys.argv[1]
#     # Load Sample_decode_job instance
#     with open(f'{counter}.pkl', 'rb') as f:
#         job = pickle.load(f)
#     # Get result back from Sample_decode_job instance
#     result = job.sample_and_print_result()
#     # Print result
#     # Just let the printed messaged be loaded into the .out files automatically.
#     json_str = json.dumps(result)
#     sys.stdout.buffer.write(json_str.encode())

# def decode_locally(counter):
#     pass


# if __name__ == '__main__':
#     main()


