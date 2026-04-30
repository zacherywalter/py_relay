# # Bivarate Bicycle Code construction
# see [BCG24] Bravyi, S., Cross, A.W., Gambetta, J.M. et al. High-threshold and low-overhead fault-tolerant quantum memory. Nature 627, 778–782 (2024). https://doi.org/10.1038/s41586-024-07107-7
# a few functions for constructing a Bivariat Bicycle code

import numpy as np
import scipy as sp
import galois
import matplotlib.pyplot as plt

GF2 = galois.GF(2)

rank = np.linalg.matrix_rank
pow = np.linalg.matrix_power

def S(d):
    return np.roll(np.eye(d),-1, axis=0)


def generate_bb(l:int,m:int,a_tuple:tuple[int, ...],b_tuple:tuple[int, ...], dw:tuple[int, ...] =(1,1,1,1,1,1)):
    """Create the two quasi cyclic matracies used to construct a bivariate bicycle code
    args:
        see generate_bb_pcms()
    """
    (a1,a2,a3) = a_tuple
    (b1,b2,b3) = b_tuple

    x = GF2(np.kron(S(l),np.eye(m)).astype(int))
    y = GF2(np.kron(np.eye(l),S(m)).astype(int))

    # TODO: are A and B also conjugate pair?
    A = dw[0]*pow(x,a1) + dw[1]*pow(y,a2) + dw[2]*pow(y,a3)
    B = dw[3]*pow(y,b1) + dw[4]*pow(x,b2) + dw[5]*pow(x,b3)
    return A,B
    
def generate_bb_pcms(l:int, m:int, a_tuple:tuple[int, ...], b_tuple:tuple[int, ...], dw:tuple[int, ...]=(1,1,1,1,1,1)):
    """Return the two Parity Check Matricies H_x and H_z that can be used to
    construct a Calderbank-Shor-Steane (CSS) code.
    args:
        l:
            number of blocks
        m:
            block size (number of bits per block)
        a_tuple, b_tuple:
            are both size 3 tuples containing the positive integer powers of x and y
            that make up the polynomials/matricies A and B
        dw:
            display weights can be used to remove or distort parts of the bb code
    """
    A,B = generate_bb(l,m,a_tuple,b_tuple, dw)
    H_x = np.concatenate((A,B),axis=1)
    H_z = np.concatenate((np.transpose(B),np.transpose(A)),axis=1)
    return H_x,H_z

def apply_chess_board(A,l,m):
    # spacing checkerboard pattern to see structure
    # https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    spacing = 0.15*(np.kron(np.indices((l,2*l)).sum(axis=0)%2,np.ones((m,m))))
    matrix = np.clip(np.array(A) + spacing[:A.shape[0],:A.shape[1]],0,1)
    return matrix


def find_logical_z(H_x, H_z): return find_logical(H_x, H_z)
def find_logical_x(H_x, H_z): return find_logical(H_z, H_x)
def find_logical(H_1, H_2):
    ##### warning: Gemini Code!!! #####
    # 1. Find the kernel of H_1 (All strings that commute with X-stabilizers)
    # This contains both Logical Z and Z-stabilizers
    G_1 = GF2.null_space(GF2(H_1)) 
    
    # 2. We need to find vectors in G_1 that are NOT in the row space of H_2
    # A common way is to use Gaussian elimination on H_2 to get a reduced row echelon form (RREF)
    # then check which rows of G_1 cannot be produced by H_2.
    
    # Alternatively, use a projection or systematic basis:
    # Combine H_2 and G_1 and find the basis of the combined matrix.
    # The vectors in G_1 that increase the rank of H_2 are your logicals.
    
    logicals = []
    current_basis = GF2(H_2)
    base_rank = np.linalg.matrix_rank(current_basis)
    
    for row in G_1:
        test_matrix = np.vstack([current_basis, row])
        if np.linalg.matrix_rank(test_matrix) > base_rank:
            logicals.append(row)
            current_basis = test_matrix
            base_rank += 1
            
    return np.array(logicals)
    

# Example [[144,12,12]]
# l = 12 # number of blocks
# m = 6 # bits per block
# a_tuple = (3,1,2)  # power of x,y,y
# b_tuple = (3,1,2)  # power of y,x,x
# 
# H_x, H_z = generate_bb_pcms(l,m,a_tuple,b_tuple)
# lz = find_logical_z(H_x, H_z)
# lx = find_logical_x(H_x, H_z)
# print(lz, lx)

# Explaination of Construction
# $\bm{S}_d$ is the cyclic shift matrix $\in\mathbb{F}_2^{d\times d}$. For example
# $\bm{S}_3 = \begin{bmatrix} 0&1&0 \\ 0&0&1 \\ 1&0&0 \end{bmatrix}$ <br>
# 
# $x = \bm{S}_l \otimes \mathbb{I}_m$ <br>
# $y = \mathbb{I}_l \otimes \bm{S}_m$ <br>
# <br>
# 
# matracies $A$ and $B$ both have size $lm \times lm$ and row weight 3$<br>
# $\bm{A} = x^{a_1} + y^{a_2} + y^{a_3}$<br>
# $\bm{B} = y^{b_1} + x^{b_2} + x^{b_3}$<br>
# <br>
# $\bm{H}^{\text{X}} = \begin{bmatrix} \bm{A} | \bm{B} \end{bmatrix}$<br>
# $\bm{H}^{\text{Z}} = \begin{bmatrix} \bm{A}^{\text{T}} | \bm{B}^{\text{T}} \end{bmatrix}$

def eliminate_dependent_rows(H):
    matrix = GF2(H)
    current_rank = 1
    reduced_matrix = GF2(matrix[0])
    for row in matrix:
        test_matrix = np.vstack([reduced_matrix, row])
        if np.linalg.matrix_rank(test_matrix) > current_rank:
            reduced_matrix = test_matrix
            current_rank += 1
    return reduced_matrix