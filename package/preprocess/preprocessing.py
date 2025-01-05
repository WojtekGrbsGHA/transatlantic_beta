import numpy as np


def _validate_null(matrix: np.ndarray):
    """
    Check if the given matrix has any missing values.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be checked.
    """
    if np.any(matrix == None):
        raise ValueError('Missing data found in the input matrix!')


def _validate_dimensions(matrix: np.ndarray):
    """
    Check if the given matrix has the same number of columns in each row.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be checked.
    """
    if matrix.ndim != 2:
        raise ValueError('Input matrix is not 2-dimensional!')

def _validate_consistent_columns(matrix: np.ndarray):
    """
    Check if the given matrix has the same number of columns in each row.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be checked.
    """
    # Get the number of columns from the first row
    num_columns = matrix.shape[1]

    for row in matrix:
        if row.size != num_columns:
            raise ValueError(f'Dimensions not consistent among all datapoints! Declared matrix dimension: {num_columns}, Outlier dimension: {row.size}')
        

def _preprocess_ndarray_matrix(matrix: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Performs basic casting to a desired dtype and performs checks if the matrix is 

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be prorcessed.
    dtype: type
        The dtype of desired output matrix.

    Returns
    ----------
    np.ndarray
        Preprocessed 
    """
    if matrix.dtype != np.float64:
        try:
            matrix = matrix.astype(np.float64)
        except:
            raise TypeError(f'Array datatype not recognised or cannot be casted to {dtype}')
        
    if not matrix.flags['C_CONTIGUOUS']:
            matrix = np.ascontiguousarray(matrix) # contiguous for rowwise speed
    _validate_null(matrix)
    _validate_dimensions(matrix)
    _validate_consistent_columns(matrix)
    return matrix

def _preprocess_labels(labels):
    pass