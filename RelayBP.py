from DMemBP import DMemBP
import numpy as np

class RelayBP(DMemBP):
    def __init__(self, check_matrix,                                   
        error_priors,  # TODO: Question: in general should i be passing llrs here???
        gamma0,
        pre_iter,
        num_sets,  # R_legs
        set_max_iter,  # max_iter_per_leg
        gamma_dist_interval,
        stop_nconv,  # n_sought_solutions
    ):
        super().__init__(check_matrix, error_priors, set_max_iter, gamma_dist_interval)
        self.gamma0 = gamma0
        self.pre_iter = pre_iter
        self.num_sets = num_sets
        self.stop_nconv = stop_nconv

        self.minimum_weight = float('inf')
        self.found_solutions = 0
        self.best_error_estimate = None
        self.relay_marginals = None
    
    def decode(self, syndrome):
        # do pre dmembp
        error_estimate, self.relay_marginals = super().decode(syndrome)
        self.minimum_weight = float('inf')
        for r in range(self.num_sets):
            # do dmembp and pass on the marginals
            error_estimate, self.relay_marginals = super().decode(syndrome, self.relay_marginals)
            estimate_weight = np.sum(error_estimate.decoding*self.error_priors).item()
            self.found_solutions += 1  # TODO: dmempbp always returns a solutions (even invalid ones)!
            if(estimate_weight < self.minimum_weight):
                self.best_error_estimate = error_estimate.decoding
                self.minimum_weight = estimate_weight
            
            if(self.found_solutions == self.stop_nconv):
                self.result.decoding = self.best_error_estimate
                return self.result
        
        self.result.decoding = self.best_error_estimate
        return self.result
        