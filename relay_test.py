from RelayBP import RelayBP

import numpy as np
import scipy as sp
import bivariate_bicycle as bb
import latex_plot
import results
import galois
import pytikz

GF2 = galois.GF(2)

## Example [[144,12,12]]
code = "BB_144_12_12"

l = 12 # number of blocks
m = 6 # bits per block
a_tuple = (3,1,2)  # power of x,y,y
b_tuple = (3,1,2)  # power of y,x,x

H_x_full, H_z_full = bb.generate_bb_pcms(l, m, a_tuple, b_tuple)
H_x = bb.eliminate_dependent_rows(H_x_full)
H_z = bb.eliminate_dependent_rows(H_z_full)

lx = bb.find_logical_x(H_x, H_z)
lz = bb.find_logical_z(H_x, H_z)

# test [7,4] hamming code
H_x = GF2(np.array([[1,0,1,0,1,0,1],
                    [0,1,1,0,0,1,1],
                    [0,0,0,1,1,1,1]]))

lx = np.array(GF2(H_x).null_space())

check_matrix = sp.sparse.csr_matrix(np.asarray(H_x))# detectors x error variables

# Decoding problem specification
# batch_size * max_n_batches > 10_000_000 is good maybe
batch_size = 10
max_n_batches = 10_000_000
max_logical_errors = 100
# error_probability = [0.008, 0.02, 0.04, 0.06, 0.08]
error_probability = [0.02, 0.04, 0.06, 0.08]

# Decoder configuration
gamma0 = 0.1                      # Uniform memory weight for the first ensemble
pre_iter = 80                      # Max BP iterations for the first ensemble 
num_sets=100                       # Number of relay ensemble elements
set_max_iter = 60                 # Max BP iterations per relay ensemble
gamma_dist_interval=(-0.24, 0.66)  # Set the uniform distribution range for disordered memory weight selection
# gamma_dist_interval=(0.1, 0.10000001)
stop_nconv=5                      # Number of relay solutions to find before stopping (the best will be selected) 

convergence_rates = []
LERs = []
max_trials = []

for p in error_probability:
    error_priors = np.repeat(np.array(p), H_x.shape[1])
    batch_count = 0
    decoding_failures = 0

    # python Relay decoder
    decoder = RelayBP(
        H_x,                                   
        error_priors=error_priors,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_interval=gamma_dist_interval,
        stop_nconv=stop_nconv,
    )

    print("p=", p)
    # run monte carlo simulation untill enough errors have been collected
    while batch_count<max_n_batches and decoding_failures<max_logical_errors:
        errors = np.array((np.random.rand(batch_size, H_x.shape[1]) < p), dtype=np.uint8)
    
        basic_detectors = np.squeeze(np.array(H_x@GF2(np.expand_dims(errors, 2)), dtype=np.uint8), axis=2)
        errors_decoded_detailed = []
        for detector in basic_detectors:
            errors_decoded_detailed.append(decoder.decode(detector))

        # basic_detectors = np.squeeze(np.array(H_x@GF2(np.expand_dims(errors, 2)), dtype=np.uint8), axis=2)
        # errors_decoded_detailed = decoder.decode_batch(basic_detectors)
    
        observable_frame_errors = []
        for i, result in enumerate(errors_decoded_detailed):
            # calculate observable errors
            logical_error = (GF2(lx)@GF2(result.decoding) != GF2(lx)@GF2(errors[i])).any()
            # logical_error = (result.decoding != errors[i]).any()
            observable_frame_errors.append(logical_error)
            
        decoding_failures += np.sum(observable_frame_errors).item()
        batch_count += 1
        if batch_count%10 == 0:
            print(".", end="")

    logical_error_rate = decoding_failures / (len(errors)*batch_count)

    LERs.append(logical_error_rate)
    max_trials.append(batch_count)
    print(f"For error rate {p}: LER = {logical_error_rate}")  # and convergence rate = {rate_converged}")

results_dict = {}
for i in range(len(LERs)):
    results_dict[error_probability[i]] = LERs[i]

# max_trials = [batch_size] * len(LERs)
LERs = np.array(LERs)
sigma = np.sqrt(LERs * (1 - LERs) / max_trials)
results.plot_FER((results_dict, "Sim"), title=code+" DMemBP", sigma=sigma)
# pytikz.addplot_table(list(error_probability), list(LERs), "py_relay")
pass