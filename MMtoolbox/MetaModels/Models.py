# Import numpy for matrices
import numpy as np
import warnings


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

        # Checks if the intervals for the input parameters is formulated correctly
        if not isinstance(in_par_intervals, np.matrix) or in_par_intervals.shape[0] != 2:
            raise TypeError('the input parameter intervals are not in a matrix of the correct size')
        if not np.all(np.isreal(in_par_intervals)):
            raise TypeError('The input parameter intervals are not only real numbers')

        # Checks if the means for the input parameters is formulated correctly
        if not isinstance(in_par_means, np.matrix) or in_par_means.shape[0] != 1:
            raise TypeError('the input parameter means are not in a matrix or of the correct size')
        if not np.all(np.isreal(in_par_means)):
            raise TypeError('The input parameter means are not only real numbers')

        # Checks if the variances for the input parameters is formulated correctly
        if not isinstance(in_par_variances, np.matrix) or in_par_variances.shape[0] != 1:
            raise TypeError('the input parameter variances are not in a matrix or of the correct size')
        if not np.all(np.isreal(in_par_variances)):
            raise TypeError('The input parameter variances are not only real numbers')

        # Checks if the all input parameter details have the same number of input parameters
        if not (in_par_intervals.shape[1] == in_par_means.shape[1] == in_par_variances.shape[1]):
            raise TypeError('different number of input parameters: %x intervals, %x means, %x variances'
                             % (in_par_intervals.shape[1], in_par_means.shape[1], in_par_variances.shape[1]))

        # Checks if interval boundaries are showing a positive interval and
        # checks the means and variances for the input parameters is formulated between their interval boundaries
        for i in range(in_par_intervals.shape[0]):
            if in_par_intervals[i, 0] > in_par_intervals[i, 1]:
                raise ValueError('The interval boundaries of input parameter %x are negative' % i)
            if in_par_means[0,i] < in_par_intervals[i, 0] or in_par_means[0, i] > in_par_intervals[i, 1]:
                raise ValueError('mean of input parameter %x are not within interval' % i)
            if 2 * pow(in_par_variances[0, i], 1 / 2) > (in_par_intervals[i, 1] - in_par_intervals[i, 0]):
                raise ValueError('variance of input parameter %x is too big for the interval' % i)

        # Checks if the intervals for the output parameters is formulated correctly
        if not isinstance(out_par_intervals, np.matrix) or out_par_intervals.shape[0] != 2:
            raise TypeError('the output parameter intervals are not in a matrix of the correct size')
        if not np.all(np.isreal(out_par_intervals)):
            raise TypeError('The output parameter intervals are not only real numbers')

        # Checks if the means for the outnput parameters is formulated correctly
        if not isinstance(out_par_means, np.matrix) or out_par_means.shape[0] != 1:
            raise TypeError('the output parameter means are not in a matrix or of the correct size')
        if not np.all(np.isreal(out_par_means)):
            raise TypeError('The output parameter means are not only real numbers')

        # Checks if the variances for the output parameters is formulated correctly
        if not isinstance(out_par_variances, np.matrix) or out_par_variances.shape[0] != 1:
            raise TypeError('the output parameter variances are not in a matrix or of the correct size')
        if not np.all(np.isreal(out_par_variances)):
            raise TypeError('The output parameter variances are not only real numbers')

        # Checks if the all output parameter details have the same number of output parameters
        if not (out_par_intervals.shape[1] == out_par_means.shape[1] == out_par_variances.shape[1]):
            raise TypeError('different number of output parameters: %x intervals, %x means, %x variances'
                             % (out_par_intervals.shape[1], out_par_means.shape[1], out_par_variances.shape[1]))

        # Checks if interval boundaries are showing a positive interval and
        # checks the means and variances for the output parameters is formulated between their interval boundaries
        for i in range(out_par_intervals.shape[0]):
            if out_par_intervals[i, 0] > out_par_intervals[i, 1]:
                raise ValueError('The interval boundaries of output parameter %x are negative' % i)
            if out_par_means[0, i] < out_par_intervals[i, 0] or out_par_means[0, i] > out_par_intervals[i, 1]:
                raise ValueError('mean of output parameter %x are not within interval' % i)
            if 2 * pow(out_par_variances[0, i], 1 / 2) > (out_par_intervals[i, 1] - out_par_intervals[i, 0]):
                raise ValueError('variance of output parameter %x is too big for the interval' % i)

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
        return self.__mm_type

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

        # Checks if the input parameters are in a list
        if not isinstance(raw_input_par, np.matrix):
            raise TypeError('The input parameters are not in a list')

        # Checks if the number of input parameters is correct
        if raw_input_par.shape[1] != self.__in_par_means.shape[1]:
            raise TypeError('The number of input parameters is different from the meta-model')

        if not np.all(np.isreal(raw_input_par)):
            raise TypeError('Input parameters are not defined as real numbers')

        # Checks if the input parameters are numbers and are in the predefined intervals
        for i in range(raw_input_par.shape[1]):
            if raw_input_par[0, i] < self.__in_par_intervals[i, 0] \
                    or raw_input_par[0, i] > self.__in_par_intervals[i, 1]:
                warnings.warn('Input parameter %x is not in between the predefined intervals' % i)

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

    __mm_type = 'PLSR'

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

        # Checks if the solution matrix is formulated correctly
        if not isinstance(sol_mat, np.matrix):
            raise TypeError('the solution matrix is not a matrix')

        # Check if the size of the solution matrix is right
        if sol_mat.shape[0] != in_par_intervals.shape[1] + 1 or sol_mat.shape[1] != out_par_intervals.shape[1]:
            raise TypeError('the solution matrix does not have the correct size')

        # Check if the values of the solution matrix are numbers
        if not np.all(np.isreal(sol_mat)):
            raise TypeError('the values solution matrix are not real numbers')

        # Add additional variables
        self.__mm_type = 'PLSR'
        self.__output_const = np.mat(sol_mat[0])
        self.__regress_coeff = np.transpose(np.mat(sol_mat[1:]))

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
        mod_in_par = raw_input_par
        return mod_in_par

    def calculate_output(self, input_par):
        """ A method that calculates the output for input parameters

        :param input_par: A list of (modified) input parameters
        :return: The corresponding output
        """

        output_var = np.transpose(self.__regress_coeff * input_par)
        output_par = self.__output_const + output_var

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
        __mm_type: The type of the meta-model (In this PLSR class, it is 'PLSR')
        __input_data: A database of different input combinations to be checked for closest resemblance new input
        __output_data: A database of output combinations linked to the input combinations database, in which the
            corresponding output is searched for
    """

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

        # Checks if the input database is formulated correctly
        if not isinstance(input_data, np.matrix):
            raise TypeError('the input database is not a matrix')

        # Checks if the input database has the correct size
        if input_data.shape[1] != in_par_intervals.shape[1]:
            raise TypeError('the input database does not have the correct size')

        # Checks if the input database contains only numbers
        if not np.all(np.isreal(input_data)):
            raise TypeError('the values of the input database are not real numbers')

        # Checks if the output database is formulated correctly
        if not isinstance(output_data, np.matrix):
            raise TypeError('the output database is not a matrix')

        # Checks if the output database has the correct size
        if output_data.shape[1] != out_par_intervals.shape[1]:
            raise TypeError('the output database does not have the correct size')

        # Checks if the output database contains only numbers
        if not np.all(np.isreal(output_data)):
            raise TypeError('the values of the output database are not real numbers')

        # Checks if the in- and output database have the same size
        if input_data.shape[0] != output_data.shape[0]:
            raise TypeError('the output database does not have the correct size')

        # Add additional variables
        self.__mm_type = 'DLU'
        self.__input_data = input_data
        self.__output_data = output_data

    def modify_input(self, raw_input_par):
        """ A method that modifies the raw input parameters ready for calculating the output

        :param raw_input_par: A list of input parameters that still need to be modified to calculate the output
        :return: The modified input parameters. In this case the same as the raw input
        """
        mod_in_par = raw_input_par
        return mod_in_par

    def calculate_output(self, input_par):
        """ A method that calculates the output for input parameters

        :param input_par: A list of (modified) input parameters
        :return: The corresponding output
        """

        def find_euc_dist(list_par1, list_par2):
            """ Finds the Euclidian distance between the two parameter points

            :param list_par1: List with the parameter values of point 1
            :param list_par2: List with the parameter values of point 2
            :return: The euclidian distance between the two values
            """

            # First find the squared Euclidian distance
            temp_euc_dist = 0
            for i in range(len(list_par1)):
                temp_euc_dist += pow(list_par1[0, i]-list_par2[0, i], 2)

            # Then return the root of this distance
            return pow(temp_euc_dist, 1/2)

        # Initialization of the closest combination index at the start
        closest_comb_index = 0

        # The database is gone through to find the combination with the lowest Euclidian distance
        for i in range(self.__input_data.shape[0]):
            new_comb_index = find_euc_dist(self.__input_data[i], input_par)

            if closest_comb_index > new_comb_index:
                closest_comb_index = new_comb_index

        # Return the output data corresponding to the closest parameter combination index
        return self.__output_data[closest_comb_index]


class ModelDecorator(AbstractModel):
    """ A decorator class for different meta-model additions based on input parameter modifiers and clustering

    Attributes:
        __mm_type: The type of the meta-model decoration (In this Decorator class, it is 'Decorated')
        __meta_model: The meta_model that is decorated
    """

    __mm_type = 'Decorated '

    def __init__(self, meta_model):
        """ The initialization fo the decorator. All attributes will be the same as the original model, with the
        additions that the old model is kept as an attribute as well

        :param meta_model: meta_model is the attribute to be decorated
        """

        self.__meta_model = meta_model

        AbstractModel.__init__(self, meta_model.get_in_par_intervals(), meta_model.get_in_par_means(),
                               meta_model.get_in_par_variances(), meta_model.get_out_par_intervals(),
                               meta_model.get_out_par_means(), meta_model.get_out_par_variances())

    def get_type(self):
        """ Returns the type of the meta-model as well as the decorator additions

        :return: decorated meta-model type
        """
        return self.__mm_type + self.__meta_model.get_type()

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


class InputDecorator(ModelDecorator):
    """ The abstract main class for all input specification decorators . It changes the input modification to a new
    by adding more.

    Attributes:
        __input_spec: The specifications of the input decorations (In this InputDecorator class, it is 'Abstract')
        __mm_type: The type of the meta-model decoration (In this InputDecorator class, it is 'Decorated Input')
    """

    # Input modifier specifications
    __input_spec = 'Abstract'
    __mm_type = 'Decorated Input '

    def modify_input(self, raw_input_par):
        """ A method to prepare the raw input parameters to parameters used for calculating the output

        :param raw_input_par: A list of the raw input parameters that have to be modified
        :return: Raises an error since this is an abstract class, subclasses have to implement this method
        """
        raise NotImplementedError

    def calculate_output(self, input_par):
        """ The other method that can be decorated. It is not decorated in this subclass.

        :param input_par: The input parameters used for calculating the output
        :return: The corresponding output parameters to the input parameters
        """
        return self.__meta_model.calculate_output(input_par)

    def get_input_spec(self):
        """ Gives the input modifier specifications of this decorator class

        :return: The input specifications
        """
        return self.__input_spec


class PolynomialDecorator(InputDecorator):
    """ The polynomial input modifier decorator class for all meta-models. This decorator cöass changes the input
    parameter by adding polynomial terms to the output

    Attributes:
        __input_spec: The specifications of the input decorations (In this PolynomialDecorator class, it is
            'Polynomial')
        __mm_type: The type of the meta-model decoration (In this PolynomialDecorator class, it is 'Polynomial Input')
    """

    __input_spec = "Polynomial"
    __mm_type = "Polynomial Input "

    def modify_input(self, raw_input_par):
        """ A method to prepare the raw input parameters to parameters used for calculating the output. In this
        decorator a polynomial version of the input is returned

        :param raw_input_par: A list of the raw input parameters that have to be modified
        :return: The input parameters with their polynomial parameters added
        """

        nr_par = raw_input_par.shape[1]
        mod_input_par = self.standardize_input(raw_input_par)

        # Add all terms of the polynomial input parameters to the modified input parameters
        for i in range(nr_par):
            for j in range(i, nr_par):
                mod_input_par[(nr_par - 1) * i + j] = raw_input_par[0, i] * raw_input_par[0, j]

        return self.__meta_model.modify_input(mod_input_par)

    def standardize_input(self, raw_input_par):
        """ A method to standardize the input parameters. This is done by substracting them by the mean and dividing
        them by the standard deviation of their parameters

        :param raw_input_par: The to be standardized input parameters
        :return: The standardized input parameters
        """

        stand_input_par = raw_input_par

        # Standardize by substracting the mean and dividing by the standarad deviations
        for i in range(raw_input_par.shape[1]):
            mean_par = raw_input_par[0, i] - self.__in_par_means[0, i]
            stand_input_par[0, i] = mean_par / pow(self.__in_par_variances[0, i], 1 / 2)

        return stand_input_par
