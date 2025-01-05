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

def _preprocess_labels(y):
    """
    Ensures that y is an iterable castable to a 1D integer NumPy array.
    
    Checks:
    1) y is not None.
    2) y is an iterable.
    3) y can be cast to an integer NumPy array.
    4) y is not empty.
    5) y has no NaN values.
    6) y is 1-dimensional.
    7) All values in y are integers in the range [-1, inf).

    Returns:
        y_arr (np.ndarray): The validated and processed y as an integer array.
    """

    # Check if y is None
    if y is None:
        raise ValueError("y cannot be None.")

    # Check if y is an iterable
    try:
        iter(y)
    except TypeError:
        raise TypeError("y must be an iterable object (e.g., list, tuple, Series).")

    # Attempt to cast y to an integer NumPy array
    try:
        y_arr = np.array(y, dtype=int)
    except ValueError:
        raise ValueError("y could not be cast to an integer array.")

    # Check if the resulting array is empty
    if y_arr.size == 0:
        raise ValueError("y must not be an empty array.")

    # Check for NaN values
    if np.isnan(y_arr).any():
        raise ValueError("y contains NaN values, which is not allowed.")

    # Check dimensionality (assuming we want a 1D array)
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1-dimensional array.")

    # Check if all values in y are integers in the range [-1, inf)
    if not np.all((y_arr >= -1)):
        raise ValueError("y contains values outside the allowed range [-1, infinity).")
    else:
        return y_arr
