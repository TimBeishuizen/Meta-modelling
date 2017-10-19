""" Several generic methods to achieve robustness in the used methods

"""

import warnings
import numpy as np
import os
from MetaModels import MetaModel as MM



def check_if_matrix(matrix, name):
    """ Checks if input parameter matrix is actually a matrix

    :param matrix: Input to be checked for a matrix
    :param name: The name of the to be checked matrix
    :return: Checks if the input is a matrix
    """

    # Check if it is a matrix
    if not isinstance(matrix, np.matrix):
        raise TypeError(name + ' is not a matrix')
    # Check if all the values in the matrix are real numbers
    elif np.isnan(matrix).any() or not np.all(np.isreal(matrix)):
        raise TypeError(name + ' does not only have real numbers as values')


def check_if_ndim_array(ndim_array, ndim, name):
    """ Checks if the ndim_array is actual an array of ndim dimensions

    :param ndim_array: The to be tested ndim-dimensional array
    :param ndim: The number of dimensions
    :param name: The name of the array
    :return: Checks if the ndim_array is actual an array of ndim dimensions
    """

    if not isinstance(ndim_array, np.ndarray):
        raise TypeError(name + ' is not an multidimensional array')
    elif np.ndim(ndim_array) != ndim:
        raise TypeError(name + ' does not have %x dimensions' % ndim)
    elif np.isnan(ndim_array).any() or not np.all(np.isreal(ndim_array)):
        raise TypeError(name + ' does not only have real numbers as values')


def check_if_same_size(size_1, size_2, name_1, name_2):
    """ Checks if two sizes are the same

    :param size_1: The first to be checked size
    :param size_2: The second to be checked size
    :param name_1: The name of the first to be checked size
    :param name_2: The name of the second to be checked size
    :return: Checks if the two sizes are the same
    """

    if size_1 != size_2:
        raise TypeError(name_1 + ' are not of the same size ' + name_2)


def check_if_bigger(size_1, size_2, name_1, name_2):
    """ Checks if the first value is bigger than the second

    :param size_1: The first to be checked size
    :param size_2: The second to be checked size
    :param name_1: The name of the first to be checked size
    :param name_2: The name of the second to be checked size
    :return: Checks if the first size is bigger than the second
    """

    if not size_1 > size_2:
        raise TypeError(name_1 + ' is not bigger than ' + name_2)


def check_if_corr_interval(interval, in_par):
    """ Checks if the mean is within the interval

        :param interval: The interval of input parameter in_par
        :param in_par: The number of the input parameter
        :return: Checks if the mean is within the interval
        """

    if interval[0, 0] > interval[0, 1]:
        raise ValueError('The lower bound of interval %x is higher than the upper bound of interval %x'
                         % (in_par, in_par))


def check_if_in_interval(interval, value, in_par, name):
    """ Checks if the value is within the interval

        :param interval: The interval of input parameter in_par
        :param value: The value of input parameter in_par
        :param in_par: The value of the input parameter
        :param name: The name of the value
        :return: Checks if the value is within the interval
        """

    if value < interval[0, 0]:
        raise ValueError(name + ' is lower than the lower bound of input parameter %x' % in_par)
    elif value > interval[0, 1]:
        raise ValueError(name + ' is higher than the upper bound of input parameter %x' % in_par)


def check_if_corr_var(interval, variance, in_par):
    """ Checks if the size of the variance is possible for the interval

        :param interval: The interval of input parameter in_par
        :param variance: The variance of input parameter in_par
        :param in_par: The number of the input parameter
        :return: Checks if the size of the variance is possible for the interval
        """

    if 2 * pow(variance, .5) > interval[0, 1] - interval[0, 0]:
        raise ValueError('The variance of input parameter %x is too big for the interval' % in_par)


def warn_if_in_interval(interval, value, in_par):
    """ Checks if the value of input parameter in_par is within the interval

        :param interval: The interval of input parameter in_par
        :param value: The checked value for input parameter in_par
        :param in_par: The number of the input parameter
        :return: Checks if the value of input parameter in_par is within the interval
        """

    if value < interval[0, 0]:
        warnings.warn('The value of input parameter %x is lower than the lower bound of input parameter %x'
                      % (in_par, in_par))
    elif value > interval[0, 1]:
        warnings.warn('The value of input parameter %x is higher than the upper bound of input parameter %x'
                      % (in_par, in_par))


def warn_if_bigger(size_1, size_2, name_1, name_2):
    """ Warns if the first value is bigger than the second

    :param size_1: The first to be checked size
    :param size_2: The second to be checked size
    :param name_1: The name of the first to be checked size
    :param name_2: The name of the second to be checked size
    :return: Warns if the first size is bigger than the second
    """

    if size_1 > size_2:
        warnings.warn(name_1 + ' is not bigger than ' + name_2)


def check_if_type(attribute, att_type, name):
    """ Checks if attribute is a meta model

    :param attribute: The to be checked attribute
    :param att_type: The type the attribute should be
    :param name: The name of the attribute
    :return: Checks if attribute is of att_type
    """

    if not isinstance(attribute, att_type):
        raise TypeError(name + ' is not of the right type')


def check_if_poss_input_spec(input_spec, i):
    """ Checks if the input specification is a known one

    :param input_spec: The input specifications
    :param i: The number of the input specification
    :return: Checks if the input specification is a known one
    """

    poss_input_spec = ['log', 'sqr', 'root', 'inv', 'exp', 'sin', 'cos', 'tan']

    if not (input_spec in poss_input_spec):
        raise ValueError('Input specification %x : ' % i + input_spec +
                         ' not in the list of available input specifications')


def check_if_meta_model(meta_model, name):
    """ Checks if the given meta model is actually a meta-model
    
    :param meta_model: The to be checked meta-model
    :param name: the name of the meta-model
    :return: The answer of the check
    """

    if not isinstance(meta_model, MM.AbstractModel):
        raise TypeError(name + ' is not a meta_model')


def check_if_string(text, name):
    """ Checks if the given string is actually a string

    :param text: The to be checked string
    :param name: the name of the string
    :return: The answer of the check
    """

    if not isinstance(text, str):
        raise TypeError(name + ' is not a string')

def check_if_file_exists(file_name, check):
    """ Checks if a file already exists
    
    :param file_name: the name of the file
    :param check: Whether it should or should not exist
    :return: answer to the check
    """

    if os.path.exists(file_name) != check:
        raise NameError(file_name + " does not exist while it should or does exist while it should not")
