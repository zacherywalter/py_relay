import relay_bp
from DMemBP import DMemBP

import numpy as np
import scipy as sp
import bivariate_bicycle as bb
import latex_plot
import results
import galois

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

## test [7,4] hamming code
H_x = GF2(np.array([[1,0,1,0,1,0,1],
                    [0,1,1,0,0,1,1],
                    [0,0,0,1,1,1,1]]))

lx = np.array(GF2(H_x).null_space())

check_matrix = sp.sparse.csr_matrix(np.asarray(H_x))# detectors x error variables

p = 0.1
error_priors = np.repeat(np.array(p), H_x.shape[1])

# Decoder configuration
gamma0 = 0.1                      # Uniform memory weight for the first ensemble
pre_iter = 2 #                       # Max BP iterations for the first ensemble 
num_sets = 1                       # Number of relay ensemble elements
set_max_iter = 0                 # Max BP iterations per relay ensemble
# gamma_dist_interval=(-0.24, 0.66)  # Set the uniform distribution range for disordered memory weight selection
gamma_dist_interval=(0.1, 0.10000001)
stop_nconv=1                      # Number of relay solutions to find before stopping (the best will be selected) 

# Relay Decoder
relay_decoder = relay_bp.RelayDecoderF32(
    check_matrix,                                   
    error_priors=error_priors,
    gamma0=gamma0,
    pre_iter=pre_iter,
    num_sets=num_sets,
    set_max_iter=set_max_iter,
    gamma_dist_interval=gamma_dist_interval,
    stop_nconv=stop_nconv,
)

# dmembp decoder
dmembp_decoder = DMemBP(
    H_x,                                   
    error_priors=error_priors,
    max_iterations=pre_iter,
    gamma_dist_interval=gamma_dist_interval
)

mismatch = 0
for i in range(128):
    error_to_test = np.array([0,0,0,0,0,1,0])
    errors = np.array((np.random.rand(H_x.shape[1]) < p), dtype=np.uint8)
    if(H_x.shape[1] == 7):
        bin_str = '0000000000' + bin(i)[2:] 
        errors = np.array([int(bin_str[-7:][j]) for j in range(7)])

    detector = np.array(H_x@GF2(errors))
    error_decoded_dmembp, _ = dmembp_decoder.decode(detector)
    error_decoded_relay = relay_decoder.decode_detailed(detector)

    # np.testing.assert_array_almost_equal(error_decoded_dmembp.decoding, error_decoded_relay.decoding)
    if(not (error_decoded_dmembp.decoding == error_decoded_relay.decoding).all()):
        print((error_decoded_dmembp.decoding == error_decoded_relay.decoding).all(), end="")
        print(" ", error_decoded_dmembp.decoding, error_decoded_relay.decoding, errors)
        pass
        mismatch += 1
    # logical_error = (result.decoding != errors[i]).any()
    # error_decoded_dmembp = dmembp_decoder.decode(detector)
    # error_decoded_relay = relay_decoder.decode_detailed(detector)
print(mismatch, " mismatches")

pass