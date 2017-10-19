import numpy as np
from MetaModels import ConstructionMethods as CM


def polynomialize_database(input_database):
    """ Polynomialize the database

    :param input_database: The to be polyniomalized database
    :return: the polynomialized database
    """

    input_means = CM.compute_means(input_database)
    input_variances = CM.compute_variances(input_database)

    polynomialized_database = polynomialize_input(input_database[0], input_means, input_variances)

    for line in input_database[1:]:
        polynomialized_database = np.append(polynomialized_database,
                                            polynomialize_input(line, input_means, input_variances), axis=0)

    return polynomialized_database


def polynomialize_input(input_database_line, input_means, input_variances):
    """ Modifies the input database to polynomial values

    :param input_database_line:  input database
    :param input_means: means of the input
    :param input_variances: variances of the input
    :return: polynomialized input line
    """

    nr_par = int(input_database_line.shape[1])
    nr_mod_par = int(2 * nr_par + (nr_par * nr_par - 1) / 2) - 1
    mod_input_par = np.mat(np.zeros(nr_mod_par))
    stand_par = standardize_input(input_database_line, input_means, input_variances)
    mod_input_par[0, range(nr_par)] = stand_par[0]

    # Add all terms of the polynomial input parameters to the modified input parameters
    next_par = nr_par
    for i in range(nr_par):
        for j in range(i, nr_par):
            mod_input_par[0, next_par] = mod_input_par[0, i] * mod_input_par[0, j]
            next_par += 1

    return mod_input_par


def standardize_input(input_database_line, input_means, input_variances):
    """ A method to standardize the input parameters. This is done by substracting them by the mean and dividing
    them by the standard deviation of their parameters

    :param input_database_line: The to be standardized input parameters
    :param input_means: the means of the input
    :param input_variances: the variances of the input
    :return: The standardized input parameters
    """

    # Standardize by substracting the mean and dividing by the standard deviations
    mean_input_par = np.subtract(input_database_line, input_means)
    input_std = np.sqrt(input_variances)
    stand_input_par = np.divide(mean_input_par, input_std)

    return stand_input_par


def modify_database(input_database, modify_specs):
    """ The modfication of hte database with the specific modify specs

    :param input_database: The database to be modified
    :param modify_specs: The specifications of hte modifications
    :return: A modified database
    """

    modified_database = modify_input(input_database[0], modify_specs)

    for line in input_database[1:]:
        modified_database = np.append(modified_database, modify_input(line, modify_specs), axis=0)

    return modified_database


def modify_input(input_line, modify_specs):
    """ Modifies a single line in a database

    :param input_line: the line to be modified
    :param modify_specs: The specifications of the modification
    :return: The modified line
    """

    def add_modifier(raw_par, input_modifier):
        """ A modifier of the input parameters

        :param raw_par: The original to be modified input parameters
        :param input_modifier: The modifiers for the parameters
        :return: Modified input parameters
        """

        if input_modifier == 'log':
            mod_par = np.log(raw_par)
        elif input_modifier == 'sqr':
            mod_par = np.square(raw_par)
        elif input_modifier == 'root':
            mod_par = np.sqrt(raw_par)
        elif input_modifier == 'inv':
            mod_par = np.power(raw_par, -1)
        elif input_modifier == 'exp':
            mod_par = np.exp(raw_par)
        elif input_modifier == 'sin':
            mod_par = np.sin(raw_par)
        elif input_modifier == 'cos':
            mod_par = np.cos(raw_par)
        elif input_modifier == 'tan':
            mod_par = np.tan(raw_par)
        else:
            raise ValueError('Not a valid modifier')

        return mod_par

    nr_par = int(input_line.shape[1])

    mod_input_line = np.mat(np.zeros(nr_par * (len(modify_specs) + 1)))
    mod_input_line[0, 0:nr_par] = input_line

    for i in range(len(modify_specs)):
        mod_input_line[0, nr_par * (i + 1):  nr_par * (i + 2)] = add_modifier(input_line, modify_specs[i])

    return mod_input_line
