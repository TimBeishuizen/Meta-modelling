# Import numpy for matrices
import numpy as np
from MetaModels import RobustnessMethods as RM


class AbstractModel:
    """ The abstract superclass of the meta-models.
    This class contains all requirements for every meta-model, which makes other meta-models easier to understand.

    Functions:
        get_mm_type: Gives the type of meta-model (in this abstract class it will be 'Abstract')
        get_in_par_intervals: Gives the intervals of the input parameters of the meta-model
        get_in_par_means: Gives the means of the input parameters of the meta-model
        get_in_par_variances: Gives the variances of the input parameter of the meta-model
        get_out_par_intervals: Gives the intervals of the output parameters of the meta-model
        get_out_par_means: Gives the means of the output parameters of the meta-model
        get_out_par_variances: Gives the variances of the output parameter of the meta-model
        modify_input: A yet to be implemented way of modifying the raw input for the meta-model
        calculate_output: A yet to be implemented way of calculating the output for the meta-model
        simulate: gives the output for a given input with the use of the meta-model.

    Attributes:
        __mm_type: The type of the meta-model (In this abstract class, it is 'Abstract')
        __in_par_intervals: The intervals of the input parameters of the meta-model
        __in_par_means: The means of the input parameters of the meta-model
        __in_par_variances: The variances of the input parameter of the meta-model
        __out_par_intervals: The intervals of the output parameters of the meta-model
        __out_par_means: The means of the output parameters of the meta-model
        __out_par_variances: The variances of the output parameter of the meta-model

    """

    mm_type = 'Abstract'

    def __init__(self, in_par_intervals, in_par_means, in_par_variances,
                 out_par_intervals, out_par_means, out_par_variances):
        """ Initialization for the abstract class meta-model. It contains all information required to produce such a
            meta-model, intervals, means and variances for both the input and output parameters

        :param in_par_intervals: Nested List, intervals (in a list of 2 floating points)
                                    for every input parameter
        :param in_par_means: List, means for every input parameter (has to be within the input parameter intervals)
        :param in_par_variances:  List, variances for every input parameter (has to be possible for the input
                                    parameter intervals)
        :param out_par_intervals: Nested List, intervals (in a list of 2 floating points)
                                    for every output parameter
        :param out_par_means:  List, means for every output parameter (has to be within the output parameter intervals)
        :param out_par_variances: List, variances for every output parameter (has to be possible for the output
                                    parameter intervals)
        """

        def check_input():
            """ Checks if the input is robust

            :return: A good robust checkup
            """
            # Checks if the intervals for the input parameters is formulated correctly
            RM.check_if_matrix(in_par_intervals, 'The input parameter intervals matrix')
            RM.check_if_same_size(in_par_intervals.shape[1], 2, 'The input parameter intervals',
                                  'two: a minimum and maximum value of the interval')

            # Checks if the means for the input parameters is formulated correctly
            RM.check_if_matrix(in_par_means, 'The input parameter means matrix')
            RM.check_if_same_size(in_par_means.shape[0], 1, 'The input parameter means',
                                  'one: the value of the mean')

            # Checks if the means for the input parameters is formulated correctly
            RM.check_if_matrix(in_par_means, 'The input parameter means matrix')
            RM.check_if_same_size(in_par_means.shape[0], 1, 'The input parameter means',
                                  'one: the value of the means')

            # Checks if the variances for the input parameters is formulated correctly
            RM.check_if_matrix(in_par_variances, 'The input parameter variances matrix')
            RM.check_if_same_size(in_par_variances.shape[0], 1, 'The input parameter variances',
                                  'one: the value of the variances')

            # Checks if the all input parameter details have the same number of input parameters
            RM.check_if_same_size(in_par_intervals.shape[0], in_par_means.shape[1],
                                  'The number of input parameters for the intervals',
                                  'as the number of input parameters for the means')
            RM.check_if_same_size(in_par_variances.shape[1], in_par_means.shape[1],
                                  'The number of input parameters for the variances',
                                  'as the number of input parameters for the means')

            for i in range(in_par_intervals.shape[0]):
                # Checks if interval boundaries are showing a positive interval
                RM.check_if_corr_interval(in_par_intervals[i], i)
                # checks the means and variances for the input parameters is formulated
                # between their interval boundaries
                RM.check_if_in_interval(in_par_intervals[i], in_par_means[0, i], i, 'The input parameter mean')
                RM.check_if_corr_var(in_par_intervals[i], in_par_variances[0, i], i)

            # Checks if the intervals for the output parameters is formulated correctly
            RM.check_if_matrix(out_par_intervals, 'The output parameter intervals matrix')
            RM.check_if_same_size(out_par_intervals.shape[1], 2, 'The output parameter intervals',
                                  'two: a minimum and maximum value of the interval')

            # Checks if the means for the output parameters is formulated correctly
            RM.check_if_matrix(out_par_means, 'The output parameter means matrix')
            RM.check_if_same_size(out_par_means.shape[0], 1, 'The output parameter means',
                                  'one: the value of the mean')

            # Checks if the means for the output parameters is formulated correctly
            RM.check_if_matrix(out_par_means, 'The output parameter means matrix')
            RM.check_if_same_size(out_par_means.shape[0], 1, 'The output parameter means',
                                  'one: the value of the means')

            # Checks if the variances for the output parameters is formulated correctly
            RM.check_if_matrix(out_par_variances, 'The output parameter variances matrix')
            RM.check_if_same_size(out_par_variances.shape[0], 1, 'The output parameter variances',
                                  'one: the value of the variances')

            # Checks if the all output parameter details have the same number of output parameters
            RM.check_if_same_size(out_par_intervals.shape[0], out_par_means.shape[1],
                                  'The number of output parameters for the intervals',
                                  'as the number of output parameters for the means')
            RM.check_if_same_size(out_par_variances.shape[1], out_par_means.shape[1],
                                  'The number of output parameters for the variances',
                                  'as the number of output parameters for the means')

            for i in range(out_par_intervals.shape[0]):
                # Checks if interval boundaries are showing a positive interval
                RM.check_if_corr_interval(out_par_intervals[i], i)
                # Checks the means and variances for the output parameters is formulated
                # between their interval boundaries
                RM.check_if_in_interval(out_par_intervals[i], out_par_means[0, i], i, 'The output parameter mean')
                RM.check_if_corr_var(out_par_intervals[i], out_par_variances[0, i], i)

        check_input()

        # Add this variables to the model
        self.__in_par_intervals = in_par_intervals
        self.__in_par_means = in_par_means
        self.__in_par_variances = in_par_variances
        self.__out_par_intervals = out_par_intervals
        self.__out_par_means = out_par_means
        self.__out_par_variances = out_par_variances
        self.__mm_type = 'Abstract'

    def get_type(self):
        """ A method to obtain the type of the meta-model

        :return: A string containing the type of meta-model
        """
        return self.mm_type

    def get_in_par_intervals(self):
        """ A method to obtain the input parameter intervals

        :return: A nested list containing the lowest and highest point of the input parameter intervals
        """
        return self.__in_par_intervals

    def get_in_par_means(self):
        """ A method to obtain the input parameter means

        :return: A list containing the means of the input parameters
        """
        return self.__in_par_means

    def get_in_par_variances(self):
        """ A method to obtain the input parameter variances

        :return: A list containing the variances of the input parameters
        """
        return self.__in_par_variances

    def get_out_par_intervals(self):
        """ A method to obtain the output parameter intervals

        :return: A nested list containing the lowest and highest point of the output parameter intervals
        """
        return self.__out_par_intervals

    def get_out_par_means(self):
        """ A method to obtain the output parameter means

        :return: A list containing the means of the output parameters
        """
        return self.__out_par_means

    def get_out_par_variances(self):
        """ A method to obtain the output parameter variances

        :return: A list containing the mariances of the output parameters
        """
        return self.__out_par_variances

    def modify_input(self, raw_input_par):
        """ A method to prepare the raw input parameters to parameters used for calculating the output

        :param raw_input_par: A list of the raw input parameters that have to be modified
        :return: Raises an error since this is an abstract class, subclasses have to implement this method
        """
        raise NotImplementedError

    def calculate_output(self, input_par):
        """ A method to calculate the output, parameters using (modified) input parameters

        :param input_par: A list of the input parameters that are used to calculate the output
        :return: Raises an error since this is an abstract class, subclasses have to implement this method
        """
        raise NotImplementedError

    def simulate(self, raw_input_par):
        """ A method to obtain the correct output parameters using the raw input parameters

        :param raw_input_par: A list of the raw input parameters that are used to calculate the output
        :return: The output corresponding to the raw input parameters
        """

        # Checks if the input parameters are in a matrix, of the right size and within the predefined intervals
        RM.check_if_matrix(raw_input_par, 'The input parameters matrix')
        RM.check_if_same_size(raw_input_par.shape[1], self.__in_par_means.shape[1], 'The number of input parameters',
                              'the number of input parameters in the meta-model')
        for i in range(self.__in_par_intervals.shape[0]):
            RM.warn_if_in_interval(self.__in_par_intervals[i], raw_input_par[0,i], i)

        # Retrieve the modified input and calculate the output
        mod_input_par = self.modify_input(raw_input_par)
        output_par = self.calculate_output(mod_input_par)
        return output_par


class PLSRMetaModel(AbstractModel):
    """ The partial least squares regression variant of the meta-models.
    This class makes a PLSR meta-model to compute basic PLSR based meta-models. This uses all the predefined functions
    in AbstractModel with the PLSR variant additions that makes modifications of this meta-model easier to understand.

    Functions: (Additional/Changed)
        get_regress_coeff: Gives the regression coefficient of the meta-model
        get_output_const: Gives the output constants that the output values are based upon
        modify_input: Modifies the raw input for a modified input. In the basic variant, nothing happens
        calculate_output: Calculate the output for modified input. In PLSR, a solution matrix is used to calculate this.

    Attributes:
        __mm_type: The type of the meta-model (In this PLSR class, it is 'PLSR')
        __output_const: The output constants as the basis for the output calculation
        __regress_coeff: The regression coefficient

    """

    mm_type = 'PLSR'

    def __init__(self, sol_mat, in_par_intervals, in_par_means, in_par_variances,
                 out_par_intervals, out_par_means, out_par_variances):
        """ Initialization for the PLSR class meta-model. It contains all information required to produce such a
            meta-model, intervals, means and variances for both the input and output parameters as well as a solution
            matrix for the output calculation

        :param sol_mat: A nested list, the solution matrix used for calculating the output. First column is the constant
                                    other columns all correspond to one of the input parameters
        :param in_par_intervals: Nested List, intervals (in a list of 2 floating points)
                                    for every input parameter
        :param in_par_means: List, means for every input parameter (has to be within the input parameter intervals)
        :param in_par_variances:  List, variances for every input parameter (has to be possible for the input
                                    parameter intervals)
        :param out_par_intervals: Nested List, intervals (in a list of 2 floating points)
                                    for every output parameter
        :param out_par_means:  List, means for every output parameter (has to be within the output parameter intervals)
        :param out_par_variances: List, variances for every output parameter (has to be possible for the output
                                    parameter intervals)
        """

        # Construct the super-class abstract model
        AbstractModel.__init__(self, in_par_intervals, in_par_means, in_par_variances,
                               out_par_intervals, out_par_means, out_par_variances)

        def check_additional_input():
            """ Checks the additional input if correct

            :return: Checks the additional input if correct
            """

            # Checks if the solution matrix is a matrix and of the correct size
            RM.check_if_matrix(sol_mat, 'The solution matrix')
            RM.check_if_bigger(sol_mat.shape[0], in_par_means.shape[1],
                               'The number of input parameters in the solution matrix',
                               'The number of input parameters in the meta-model')
            RM.check_if_bigger(sol_mat.shape[1], out_par_means.shape[1] - 1,
                               'The number of output parameters in the solution matrix',
                               'The number of output parameters in the meta-model minus 1')
            RM.warn_if_bigger(sol_mat.shape[0], in_par_means.shape[1] + 1,
                              'The number of input parameters in the solution matrix',
                              'The number of input parameters in the meta-model')
            RM.warn_if_bigger(sol_mat.shape[1], out_par_means.shape[1],
                              'The number of output parameters in the solution matrix',
                              'The number of output parameters in the meta-model')

        check_additional_input()

        # Add additional variables
        self.__output_const = np.mat(sol_mat[0])
        self.__regress_coeff = np.mat((sol_mat[1:]))

    def get_regress_coeff(self):
        """ A method to obtain the regression coefficients

        :return: The regressions coefficients
        """

        return self.__regress_coeff

    def get_output_const(self):
        """ A method to obtain the output constants

        :return: The output constants
        """
        return self.__output_const

    def modify_input(self, raw_input_par):
        """ A method that modifies the raw input parameters ready for calculating the output

        :param raw_input_par: A list of input parameters that still need to be modified to calculate the output
        :return: The modified input parameters. In this case the same as the raw input
        """

        if raw_input_par.shape[0] > 1:
            mod_in_par = np.transpose(raw_input_par)
        else:
            mod_in_par = raw_input_par

        return mod_in_par

    def calculate_output(self, input_par):
        """ A method that calculates the output for input parameters

        :param input_par: A list of (modified) input parameters
        :return: The corresponding output
        """

        # Check if the input parameters and the solution matrix have the same size
        RM.check_if_same_size(input_par.shape[1], self.__regress_coeff.shape[0], 'The number of input parameters',
                              'the number of input parameter for the solution matrix')

        output_var = input_par * self.__regress_coeff
        output_par = np.add(self.__output_const, output_var)

        return output_par


class DLUMetaModel(AbstractModel):
    """ The direct look-up variant of the meta-models. This class makes a direct look-up meta-model to compute basic
    direct look-up based meta-models. This uses all the predefined functions in AbstractModel with the DLU variant
    additions that makes modifications of this meta-model easier to understand.

    Functions: (Additional/Changed)
        modify_input: Modifies the raw input for a modified input. In the basic variant, nothing happens
        calculate_output: Calculate the output for modified input. In DLU a database is scanned for the closest solution
            after which the respective output is returned

    Attributes:
        __mm_type: The type of the meta-model (In this DLU class, it is 'DLU')
        __input_data: A database of different input combinations to be checked for closest resemblance new input
        __output_data: A database of output combinations linked to the input combinations database, in which the
            corresponding output is searched for
    """

    mm_type = 'DLU'

    def __init__(self, input_data, output_data, in_par_intervals, in_par_means, in_par_variances,
                 out_par_intervals, out_par_means, out_par_variances):
        """ Initialization for the PLSR class meta-model. It contains all information required to produce such a
            meta-model, intervals, means and variances for both the input and output parameters as well as a solution
            matrix for the output calculation

        :param input_data: A database with several different input parameter combinations
        :param output_data: A database with output parameter combinations linked to one specific input parameter
                                    combination
        :param in_par_intervals: Nested List, intervals (in a list of 2 floating points)
                                    for every input parameter
        :param in_par_means: List, means for every input parameter (has to be within the input parameter intervals)
        :param in_par_variances:  List, variances for every input parameter (has to be possible for the input
                                    parameter intervals)
        :param out_par_intervals: Nested List, intervals (in a list of 2 floating points)
                                    for every output parameter
        :param out_par_means:  List, means for every output parameter (has to be within the output parameter intervals)
        :param out_par_variances: List, variances for every output parameter (has to be possible for the output
                                    parameter intervals)
        """

        # Initialize the abstract super class
        AbstractModel.__init__(self, in_par_intervals, in_par_means, in_par_variances,
                               out_par_intervals, out_par_means, out_par_variances)

        def check_additional_input():

            # Checks if the databases is formulated correctly
            RM.check_if_matrix(input_data, 'The input database')
            RM.check_if_matrix(output_data, 'The output database')
            RM.check_if_bigger(input_data.shape[1], in_par_intervals.shape[1] - 1,
                               'The number of input lines in the database', 'The number of input parameters - 1')
            RM.warn_if_bigger(input_data.shape[1], in_par_intervals.shape[1],
                              'The number of input lines in the database', 'The number of input parameters')
            RM.check_if_bigger(output_data.shape[1], out_par_intervals.shape[1] - 1,
                               'The number of output lines in the database', 'The number of output parameters - 1')
            RM.warn_if_bigger(output_data.shape[1], out_par_intervals.shape[1],
                              'The number of output lines in the database', 'The number of output parameters')
            RM.check_if_same_size(input_data.shape[0], output_data.shape[0],
                                  'The number of entries in the input database',
                                  'the number of entries in the output database')

        check_additional_input()

        # Add additional variables
        self.__mm_type = 'DLU'
        self.__input_data = input_data
        self.__output_data = output_data

    def modify_input(self, raw_input_par):
        """ A method that modifies the raw input parameters ready for calculating the output

        :param raw_input_par: A list of input parameters that still need to be modified to calculate the output
        :return: The modified input parameters. In this case the same as the raw input
        """
        if raw_input_par.shape[0] > 1:
            mod_in_par = np.transpose(raw_input_par)
        else:
            mod_in_par = raw_input_par

        return mod_in_par

    def calculate_output(self, input_par):
        """ A method that calculates the output for input parameters

        :param input_par: A list of (modified) input parameters
        :return: The corresponding output
        """

        # Tests if the number of input parameters is the same as the input database
        RM.check_if_same_size(input_par.shape[1], self.__input_data.shape[1],
                              'The number of input parameters', 'The number of input parameters in the input database')

        def find_euc_dist(list_par1, list_par2):
            """ Finds the Euclidian distance between the two parameter points

            :param list_par1: List with the parameter values of point 1
            :param list_par2: List with the parameter values of point 2
            :return: The euclidian distance between the two values
            """

            # First find the squared Euclidian distance
            temp_euc_dist = np.subtract(list_par1, list_par2)
            euc_dist = np.sum(np.square(temp_euc_dist))
            return pow(euc_dist, .5)

        # Initialization of the closest combination index at the start
        closest_comb_index = find_euc_dist(self.__input_data[1], input_par)
        index = 0

        # The database is gone through to find the combination with the lowest Euclidian distance
        for i in range(self.__input_data.shape[0]):
            new_comb_index = find_euc_dist(self.__input_data[i], input_par)

            if closest_comb_index > new_comb_index:
                closest_comb_index = new_comb_index
                index = i

        # Return the output data corresponding to the closest parameter combination index
        return self.__output_data[index]


