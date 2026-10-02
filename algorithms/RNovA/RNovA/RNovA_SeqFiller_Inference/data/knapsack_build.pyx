# cython: boundscheck=False, wraparound=False, nonecheck=True
import numpy as np
import torch
cimport numpy as np

def knapsack_build_vsize32(np.ndarray[np.float64_t, ndim=1] candidate_mass_cpu,
                            float mass_max, float aa_resolution):
    """
    Cython implementation of knapsack matrix builder using uint32 to represent amino acid states.
    Args:
        candidate_mass_cpu: 1D float32 array of candidate masses
        mass_max: upper bound of mass
        aa_resolution: resolution multiplier
    Returns:
        torch.Tensor of shape (mass_upperbound,)
    """
    cdef int vocab_size = candidate_mass_cpu.shape[0]
    cdef int mass_upperbound = int(round(mass_max * aa_resolution))
    
    # Convert candidate amino acid masses to integers with the resolution applied
    cdef np.ndarray[np.int64_t, ndim=1] candidate_aa_mass = \
        np.round(candidate_mass_cpu * aa_resolution).astype(np.int64)
    
    # Initialize the knapsack matrix with uint32 (1D array, each element can store 32 states)
    cdef np.ndarray[np.uint32_t, ndim=1] knapsack_matrix = \
        np.zeros(mass_upperbound, dtype=np.uint32)
    
    cdef int aa_id, target_col, i

    # First column update (col = 0)
    for aa_id in range(vocab_size):
        target_col = candidate_aa_mass[aa_id]
        if target_col < mass_upperbound:
            # Using bitwise OR to combine states for the target column
            knapsack_matrix[target_col] |= (1 << aa_id)

    # Iterate through each column and propagate the mass combinations
    for col in range(1, mass_upperbound):
        now_col = knapsack_matrix[col]
        if now_col:
            for aa_id in range(vocab_size):
                target_col = col + candidate_aa_mass[aa_id]
                if target_col < mass_upperbound:
                    # Using bitwise OR to propagate the state to target columns
                    knapsack_matrix[target_col] |= now_col
                    knapsack_matrix[target_col] |= (1 << aa_id)

    return knapsack_matrix.astype(np.uint64)

def knapsack_build_vsize64(np.ndarray[np.float64_t, ndim=1] candidate_mass_cpu,
                            float mass_max, float aa_resolution):
    """
    Cython implementation of knapsack matrix builder using uint32 to represent amino acid states.
    Args:
        candidate_mass_cpu: 1D float32 array of candidate masses
        mass_max: upper bound of mass
        aa_resolution: resolution multiplier
    Returns:
        torch.Tensor of shape (mass_upperbound,)
    """
    cdef int vocab_size = candidate_mass_cpu.shape[0]
    cdef int mass_upperbound = int(round(mass_max * aa_resolution))
    
    # Convert candidate amino acid masses to integers with the resolution applied
    cdef np.ndarray[np.int64_t, ndim=1] candidate_aa_mass = \
        np.round(candidate_mass_cpu * aa_resolution).astype(np.int64)
    
    # Initialize the knapsack matrix with uint64 (1D array, each element can store 64 states)
    cdef np.ndarray[np.uint64_t, ndim=1] knapsack_matrix = \
        np.zeros(mass_upperbound, dtype=np.uint64)
    
    cdef int aa_id, target_col, i

    # First column update (col = 0)
    for aa_id in range(vocab_size):
        target_col = candidate_aa_mass[aa_id]
        if target_col < mass_upperbound:
            # Using bitwise OR to combine states for the target column
            knapsack_matrix[target_col] |= (1 << aa_id)

    # Iterate through each column and propagate the mass combinations
    for col in range(1, mass_upperbound):
        now_col = knapsack_matrix[col]
        if now_col:
            for aa_id in range(vocab_size):
                target_col = col + candidate_aa_mass[aa_id]
                if target_col < mass_upperbound:
                    # Using bitwise OR to propagate the state to target columns
                    knapsack_matrix[target_col] |= now_col
                    knapsack_matrix[target_col] |= (1 << aa_id)

    return knapsack_matrix

def next_aa_mask_builder(np.ndarray[np.uint64_t, ndim=1] mask_segment, vocab_size: int) -> np.ndarray:
    """
    Return True if any bit is 1 in mask_segment (1D uint32 array).
    Equivalent to: (mask_segment[0] | mask_segment[1] | ...) != 0
    """
    cdef Py_ssize_t i, n = mask_segment.size
    cdef np.uint64_t acc = 0
    cdef np.ndarray[np.npy_bool, ndim=1] amino_acid_mask
    amino_acid_mask = np.zeros(vocab_size, dtype=np.bool_)

    for i in range(n):
        acc |= mask_segment[i]
    
    for i in range(vocab_size):
        amino_acid_mask[i] = (acc >> i) & 1

    return amino_acid_mask