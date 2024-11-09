from FlexibleQECSim.circuit_builder import *
from FlexibleQECSim.error_model import *

import time

@dataclass
class MCSampleDecodeJob:
    job_id: str
    circuit_id: str
    d: int
    p_e: float
    p_p: float
    shots: int
    biased_erasure: bool = True

    def sample_and_print_result(self,print_progress = False):
        if print_progress:
            from IPython.display import clear_output

        after_cz_error_model = get_2q_error_model(p_e=self.p_e,
                                                  p_p=self.p_p,
                                                  biased=self.biased_erasure)
        builder = easure_circ_builder(rounds = self.d,
                                      distance= self.d,
                                      after_cz_error_model=after_cz_error_model,
                                      measurement_error=0
                                      )
        builder.generate_helper()
        builder.gen_erasure_conversion_circuit()

        sampler = builder.erasure_circuit.compile_sampler() #expensive step, 16s for d13, 4s for d11, 0.7s for d9
        meas_samples = sampler.sample(shots=self.shots)
        converter = builder.erasure_circuit.compile_m2d_converter() #expensive step, 16s for d13, 4s for d11, 0.7s for d9
        det_samples, actual_obs_chunk = converter.convert(measurements=meas_samples,
                                                                separate_observables=True)
        t1 = time.time()
        # Decode
        new_circ_num_errors = 0
        # normal_circ_num_errors = 0
        for i in range(self.shots):
            predicted  = builder.decode_by_generate_new_circ(det_samples[i],'S',meas_samples[i])
            new_circ_num_errors += actual_obs_chunk[i][0] != predicted

            # predicted = builder.decode_without_changing_weights(det_samples[i],'S',meas_samples[i])
            # normal_circ_num_errors += actual_obs_chunk[i][0] != predicted

            if i%10 == 0 and print_progress:
                clear_output(wait=True)
                print(f'decoding finished {100*i/self.shots}%')
        t2 = time.time()
        if print_progress:
            print(f"{(t2-t1)/self.shots} per shot (d = {self.d})")
        result = {
            'job_id': self.job_id,
            'circuit_id': self.circuit_id,
            'd': self.d,
            'p_e':self.p_e,
            'p_p':self.p_p,
            'shots':self.shots,
            'new_circ':int(new_circ_num_errors),
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


