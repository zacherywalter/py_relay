import numpy as np
from ldpc.bp_decoder import BpDecoder
import warnings
warnings.filterwarnings('error')

class Result:
    def __init__(self):
        self.decoding = np.array([])
    

class DMemBP:
    def __init__(self, H, error_priors, max_iterations:int=10, gamma_dist_interval:tuple=(-0.24, 0.66), marginals = None):
        """
        args:
            H:
                parity check matrix. could be from circuit level noise or just H_x or H_z of the code
            error_priors:
                error priors: error of length n
            max_iterations:
                maximum number of iterations
            gamma_dist_interval:
                uniform distribution of memory_strengths withing gamma distance interval.
                see: 10.48550/arXiv.2506.01779 Improved belief propagation is sufficient for real-time decoding of quantum memory
        """
        self.H = np.array(H)

        # integers
        self.m = self.H.shape[0]
        self.n = self.H.shape[1]
        self.k = self.n - self.m
        self.K = 25

        # self.minimum_weight = float('inf')
        # self.found_solutions = 0

        # all length n vectors
        self.error_priors = error_priors
        self.priors_llr = np.log((1-self.error_priors)/self.error_priors)
        self.marginals = self.priors_llr
        self.bias_lambda = self.priors_llr

        self.best_error_estimate = None

        # other
        self.max_iterations = max_iterations
        self.gamma_dist_interval = gamma_dist_interval  # tuple
        low, high = self.gamma_dist_interval
        self.explicit_gammas = np.random.uniform(low, high, (self.n))

        self.result = Result()
        
        # functions
        # self.bp = BpDecoder(H, error_priors,
        #                     max_iter=max_iterations,
        #                     bp_method="minimum_sum",
        #                     ms_scaling_factor=0.75,
        #                     schedule="parallel")
        # self.bp.omp_thread_count
        
    
    def decode(self, syndrome, marginals=None):
        """
        args:
            syndrome:
                shape (m,)
        """
        if(marginals is not None): self.marginals=marginals
        else: self.marginals = self.priors_llr
        self.bias_lambda = self.priors_llr
        vn_messages = self.expand_variables(self.priors_llr)
        for i in range(self.max_iterations):
            self.update_lambda()  # dep. explicit_gammas, marginals, bias_lambda
            cn_messages = self.cn_fn(vn_messages, syndrome)
            vn_messages = self.vn_fn(cn_messages)  # dep. self.bias_lambda
            self.update_marginals(cn_messages)  # dep. bias_lambda
            error_estimate = self.hard_decision()

            # check convergence
            if((self.H@error_estimate%2 == syndrome).all()):  # bp converged
                self.result.decoding = error_estimate
                return self.result, self.marginals, True  # converged
                
        self.result.decoding = np.ones(self.n, dtype=np.uint8)# error_estimate  # np.zeros(self.n, dtype=np.uint8)#
        return self.result, self.marginals, False  # not converged
    
    def update_lambda(self):
        self.bias_lambda = self.explicit_gammas*self.priors_llr + (1-self.explicit_gammas)*self.bias_lambda
    
    def expand_checks(self, array_m):
        """take a 1-d np.array of length m and repeat it n times
        then multiply by H. Return the resulting array of shape (m,n)

        take 2-d np.array of shape (batch_size, m) and repeat in n times
        then multiply by H. Return shape (batch_size, m, n)"""
        if(array_m.ndim == 1):
            return np.multiply(np.repeat(np.expand_dims(array_m, axis=1), self.n, axis=1), self.H)
        elif(array_m.ndim == 2):
            return np.multiply(np.repeat(np.expand_dims(array_m, axis=2), self.n, axis=2), self.H)
        else:
            raise TypeError("array_m should be of dimension 1 or 2")
    
    def expand_variables(self, array_n):
        """take a 1-d np.array of length n and repeat it m times
        then multiply by H. Return the resulting array of shape (m,n)
        take 2-d np.array of shape (batch_size, n) and repeat in m times
        then multiply by H. Return shape (batch_size, m, n)"""
        if(array_n.ndim == 1):
            return np.multiply(np.repeat(np.expand_dims(array_n, axis=0), self.m, axis=0), self.H)
        elif(array_n.ndim == 2):
            return np.multiply(np.repeat(np.expand_dims(array_n, axis=1), self.m, axis=1), self.H)
        else:
            raise TypeError("array_n should be of dimension 1 or 2")
            
    def cn_fn(self, vn_messages, syndrome):
        """check node update function
        args:
            vn_messages:
                shape (m, n) array with messages at the ones
            syndrome:
                shape (m,)
        """
        sign = np.sign(vn_messages + 1e-9)  # TODO: maybe this will cause problems
        sign_prod = np.prod(sign, axis=1)
        sign_prod_expanded = self.expand_checks(sign_prod) * sign

        syndrome_expanded = self.expand_checks((-1)**np.array(syndrome, dtype=float))

        # find minimum message for each variable node
        min_messages = np.zeros_like(vn_messages)
        pre_mask = vn_messages.copy()
        pre_mask[pre_mask == 0] = float('inf')
        for j, column in enumerate(vn_messages.transpose()):
            mask = pre_mask.copy()
            mask[:,j] = float('inf')
            min_messages[:,j] = np.min(np.abs(mask), axis=1)
        min_messages *= self.H  # maybe weird with float('inf') * 0

        cn_messages = sign_prod_expanded * syndrome_expanded * min_messages
        return cn_messages

    def vn_fn(self, cn_messages):  # dep. self.bias_lambda
        # equation 2
        lambda_expanded = self.expand_variables(self.bias_lambda)
        vn_messages = lambda_expanded + self.expand_variables(np.sum(cn_messages, axis=0)) - cn_messages
        vn_messages = np.clip(vn_messages, -self.K, self.K)
        return vn_messages

    def update_marginals(self, cn_messages):
        # equation 3
        self.marginals = np.sum(cn_messages, axis=0) + self.bias_lambda
        
    def hard_decision(self):
        error_estimate = np.array(self.marginals < 0, dtype=int)
        return error_estimate


#############################################################################
################################ BATCH METHODS ##############################
#############################################################################


    def decode_batch(self, syndrome_batch, marginals=None):
        """
        args:
            syndrome_batch:
                shape (batch_size, m)
        """
        if(marginals != None): self.marginals=marginals
        self.batch_size, m = syndrome_batch.shape
        self.bias_lambda_batch = self.expand_to_batch(self.priors_llr, self.batch_size)
        self.marginals_batch = self.expand_to_batch(self.priors_llr, self.batch_size)
        vn_messages_batch = self.expand_to_batch(self.expand_variables(self.priors_llr), self.batch_size)
        self.minimum_weight = float('inf')

        not_converged = np.repeat([True], self.batch_size)
        error_estimate = np.zeros((self.batch_size, self.n))

        for i in range(self.max_iterations):
            self.update_lambda_batch()
            cn_messages_batch = self.cn_fn_batch(vn_messages_batch, syndrome_batch)
            vn_messages_batch = self.vn_fn_batch(cn_messages_batch)
            self.update_marginals_batch(cn_messages_batch)
            error_estimate[not_converged,:] = self.hard_decision_batch()[not_converged,:]

            not_converged = ((self.H@error_estimate.transpose()).transpose() != syndrome_batch).any(axis=1)
        return error_estimate
    
    def update_lambda_batch(self):
        explicit_gammas_batch = self.expand_to_batch(self.explicit_gammas, self.batch_size)
        self.bias_lambda_batch = explicit_gammas_batch*self.marginals_batch + (1-explicit_gammas_batch)*self.bias_lambda_batch

    def expand_to_batch(self, array_H, batch_size):
        """np.array of shape (m, n) becomes array of shape (batch_size, m, n)"""
        return np.repeat(np.expand_dims(array_H, axis=0), batch_size, axis=0)

    def cn_fn_batch(self, vn_messages_batch, syndrome_batch):
        """check node update function
        args:
            vn_messages_batch:
                shape (batch_size, m, n) array with messages at the ones of H
            syndrome:
                shape (batch_size, m)
        """
        sign = np.sign(vn_messages_batch + 1e-9)  # TODO: maybe this will cause problems
        sign_prod = np.prod(sign, axis=2)
        sign_prod_expanded = self.expand_checks(sign_prod) * sign

        syndrome_expanded = self.expand_checks((-1)**np.array(syndrome_batch, dtype=float))

        # find minimum message for each variable node
        min_messages = np.zeros_like(vn_messages_batch)
        pre_mask = vn_messages_batch.copy()
        pre_mask[pre_mask == 0] = float('inf')
        for j, column in enumerate(vn_messages_batch.transpose((1,0,2))):
            mask = pre_mask.copy()
            mask[:,:,j] = float('inf')
            min_messages[:,:,j] = np.min(np.abs(mask), axis=2)
        min_messages *= self.H  # maybe weird with float('inf') * 0

        cn_messages_batch = sign_prod_expanded * syndrome_expanded * min_messages
        return cn_messages_batch

    def vn_fn_batch(self, cn_messages_batch):  # dep. self.bias_lambda
        # equation 2
        vn_messages_batch = self.expand_variables(self.bias_lambda_batch) + self.expand_variables(np.sum(cn_messages_batch, axis=1)) - cn_messages_batch
        vn_messages_batch = np.clip(vn_messages_batch, -self.K, self.K)
        return vn_messages_batch
        
    def update_marginals_batch(self, cn_messages_batch):
        # equation 3
        self.marginals_batch = np.sum(cn_messages_batch, axis=1) + self.bias_lambda_batch
        
    def hard_decision_batch(self):
        error_estimate_batch = np.array(self.marginals_batch < 0, dtype=int)
        return error_estimate_batch

# errors_decoded_detailed = relay_decoder.decode_detailed_batch(basic_detectors)


def run_tests():
    import galois
    GF2 = galois.GF(2)
    H = np.array([[1,0,1,0,1,0,1],
                  [0,1,1,0,0,1,1],
                  [0,0,0,1,1,1,1]])
    G = GF2(H).null_space()

    error = np.array([0,0,1,0,0,0,0])
    syndrome = H@error%2
    error_priors = np.array([0.01]*H.shape[1])
    priors_llr = np.array([7.,-6.,5.,-4.,3.,-2.,1.])
    dmembp_decoder = DMemBP(H, error_priors)

    ######## cn_fn test ######## 
    vn_messages = dmembp_decoder.expand_variables(priors_llr)
    # vn_messages = np.array([[ 7., -0.,  5., -0.,  3., -0.,  1.],
    #                         [ 0., -6.,  5., -0.,  0., -2.,  1.],
    #                         [ 0., -0.,  0., -4.,  3., -2.,  1.]])

    cn_messages = dmembp_decoder.cn_fn(vn_messages, syndrome)
    cn_messages_expected = np.array([[-1., 0.,-1., 0.,-1., 0.,-3.],
                                     [ 0., 1.,-1., 0., 0., 1.,-2.],
                                     [ 0., 0., 0.,-1., 1.,-1., 2.]])
    np.testing.assert_array_almost_equal(cn_messages, cn_messages_expected)

    ######## vn_fn test ######## 
    dmembp_decoder.bias_lambda = np.array([1.,2.,3.,4.,5.,6.,7.])
    vn_messages = dmembp_decoder.vn_fn(cn_messages_expected)
    vn_messages_expected = np.array([[ 1., 0., 2., 0., 6., 0., 7.],
                                     [ 0., 2., 2., 0., 0., 5., 6.],
                                     [ 0., 0., 0., 4., 4., 7., 2.],])
    np.testing.assert_array_almost_equal(vn_messages, vn_messages_expected)

    pass


if __name__ == "__main__":
    run_tests()