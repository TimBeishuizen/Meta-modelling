# Import numpy for matrices
import warnings
import numpy as np
from MetaModels import MetaModel as MM


class ModelDecorator(MM.AbstractModel):
    """ A decorator class for different meta-model additions based on input parameter modifiers and clustering

    Attributes:
        __mm_type: The type of the meta-model decoration (In this Decorator class, it is 'Decorated')
        meta_model: The meta_model that is decorated
    """

    meta_model = None
    mm_type = 'Decorated Abstract '

    def __init__(self, meta_model):
        """ The initialization fo the decorator. All attributes will be the same as the original model, with the
        additions that the old model is kept as an attribute as well

        :param meta_model: meta_model is the attribute to be decorated
        """

        if not isinstance(meta_model, MM.AbstractModel):
            raise TypeError('the input argument is not a proper meta-model')

        self.meta_model = meta_model

        MM.AbstractModel.__init__(self, meta_model.get_in_par_intervals(), meta_model.get_in_par_means(),
                               meta_model.get_in_par_variances(), meta_model.get_out_par_intervals(),
                               meta_model.get_out_par_means(), meta_model.get_out_par_variances())

    def get_type(self):
        """ Returns the type of the meta-model as well as the decorator additions

        :return: decorated meta-model type
        """
        return self.mm_type + self.meta_model.get_type()

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
    input_spec = 'Abstract'
    mm_type = 'Decorated Input '

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

        return self.meta_model.calculate_output(input_par)

    def get_input_spec(self):
        """ Gives the input modifier specifications of this decorator class

        :return: The input specifications
        """
        return self.input_spec


class PolynomialInputDecorator(InputDecorator):
    """ The polynomial input modifier decorator class for all meta-models. This decorator class changes the input
    parameter by adding polynomial terms to the output

    Attributes:
        input_spec: The specifications of the input decorations (In this PolynomialInputDecorator class, it is
            'Polynomial')
        mm_type: The type of the meta-model decoration (In this PolynomialInputDecorator class, it is 'Polynomial Input')
    """

    input_spec = 'Polynomial'
    mm_type = 'Polynomial Input '

    def modify_input(self, raw_input_par):
        """ A method to prepare the raw input parameters to parameters used for calculating the output. In this
        decorator a polynomial version of the input is returned

        :param raw_input_par: A list of the raw input parameters that have to be modified
        :return: The input parameters with their polynomial parameters added
        """

        if raw_input_par.shape[0] > 1:
            raw_input_par = np.transpose(raw_input_par)

        nr_par = int(raw_input_par.shape[1])
        mod_input_par = np.mat(np.zeros((pow(nr_par, 2) + 3 * nr_par) / 2))
        stand_par = self.standardize_input(raw_input_par)
        mod_input_par[0, range(nr_par)] = stand_par[0]

        # Add all terms of the polynomial input parameters to the modified input parameters
        for i in range(nr_par):
            for j in range(i, nr_par):
                mod_input_par[0, nr_par * (i + 1) + j - i] = mod_input_par[0, i] * mod_input_par[0, j]

        return self.meta_model.modify_input(mod_input_par)

    def standardize_input(self, raw_input_par):
        """ A method to standardize the input parameters. This is done by substracting them by the mean and dividing
        them by the standard deviation of their parameters

        :param raw_input_par: The to be standardized input parameters
        :return: The standardized input parameters
        """

        # Standardize by substracting the mean and dividing by the standard deviations
        mean_input_par = np.subtract(raw_input_par, self.meta_model.get_in_par_means())

        input_std = np.sqrt(self.meta_model.get_in_par_variances())

        stand_input_par = np.divide(mean_input_par, input_std)

        return stand_input_par


class ModifiedInputDecorator(InputDecorator):
    """ The modified input modifier decorator class for all meta-models. This decorator class changes the input
    parameter by adding predefined modifier terms to the output

    Attributes:
        input_spec: A list with the specifications of the input decorations. The terms for the input to be modified
        are logarithm (log), squared (sqr), square root (root), inverse (inv), exponential (exp), sine (sin), cosine (cos), tangents (tan)
        mm_type: The type of the meta-model decoration (In this ModifiedInputDecorator class, it is 'Modified Input')
    """

    mm_type = 'Modified Input '

    def __init__(self, meta_model, input_spec):
        """ Initialization of the ModifiedInputDecorator class

        :param meta_model: The original meta-model that needs input modification
        :param input_spec: The specifications of the input modification
        """
        if not isinstance(input_spec, list):
            raise TypeError('The input specifications are not in a matrix')

        if not all(isinstance(spec, str) for spec in input_spec):
            raise TypeError('The input specifications are not strings')

        poss_input_spec = ['log', 'sqr', 'root', 'inv', 'exp', 'sin', 'cos', 'tan']

        for spec in input_spec:
            if spec not in poss_input_spec:
                raise ValueError("The input specification (" + spec + ") is not one of the predefined kinds: 'log', "
                                                                     "'sqr', 'root', 'inv', 'exp', 'sin', 'cos', 'tan'")

        InputDecorator.__init__(self, meta_model)

        self.input_spec = input_spec

    def modify_input(self, raw_input_par):
        """ A method to prepare the raw input parameters to parameters used for calculating the output. In this
        decorator a polynomial version of the input is returned

        :param raw_input_par: A list of the raw input parameters that have to be modified
        :return: The input parameters with their polynomial parameters added
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

        nr_par = int(raw_input_par.shape[1])

        mod_input_par = np.mat(np.zeros(nr_par * (len(self.input_spec) + 1)))
        mod_input_par[0, 0:nr_par] = raw_input_par

        for i in range(len(self.input_spec)):
            mod_input_par[0, 2*(i+1):  2*(i+1) + nr_par] = add_modifier(raw_input_par, self.input_spec[i])

        return self.meta_model.modify_input(mod_input_par)


class ClusterDecorator(ModelDecorator):
    """ A decorator class for different meta-model additions based on clustering

    Attributes:
        mm_type: The type of the meta-model decoration (In this Decorated Cluster class, it is 'Decorated Cluster ')
        meta_model: The meta_model that is decorated
        clust_cent: The locations of the cluster centers per input parameter
        model_data: A 3-dimensional array with the solution matrices for all different clusters
    """

    meta_model = None
    mm_type = 'Decorated Cluster '

    def __init__(self, meta_model, clust_cent, model_data):
        """ The initialization fo the decorator. All attributes will be the same as the original model, with the
        additions that the old model is kept as an attribute as well

        :param meta_model: meta_model is the attribute to be decorated
        :param clust_cent: The center points
        """

        self.meta_model = meta_model

        MM.AbstractModel.__init__(self, meta_model.get_in_par_intervals(), meta_model.get_in_par_means(),
                               meta_model.get_in_par_variances(), meta_model.get_out_par_intervals(),
                               meta_model.get_out_par_means(), meta_model.get_out_par_variances())

        # Check if the cluster center input is correct
        if not isinstance(clust_cent, np.matrix) or not np.all(np.isreal(clust_cent)):
            raise TypeError('The cluster centers are not stored in a matrix with only real numbers')

        if meta_model.get_in_par_means().shape[1] < clust_cent.shape[1]:
            warnings.warn('The number of input parameters for the cluster centers is bigger than the actual number of '
                          'input parameters')

        elif meta_model.get_in_par_means().shape[1] > clust_cent.shape[1]:
            raise TypeError('The number of input parameters for a cluster is smaller than the actual numbers of input'
                            'parameters')

        bounds = meta_model.get_in_par_intervals()

        for j in range(clust_cent.shape[0]):
            for i in range(clust_cent.shape[1]):
                if clust_cent[j, i] > bounds[i, 1] or clust_cent[j, i] < bounds[i, 0]:
                    raise ValueError('The cluster center parameters are not withing the input parameter intervals')

        # Check if the additional data is correct
        if meta_model.get_type() == 'PLSR':  # Additional check-up for PLSR

            if not isinstance(model_data, np.ndarray) or np.ndim(model_data) != 3 or not np.all(np.isreal(
                            model_data)):
                raise TypeError('The cluster solution matrices are not stored in a 3 dimensional array with only real '
                                'numbers')

            if meta_model.get_in_par_means().shape[1] < model_data.shape[1] - 1:
                warnings.warn('The number of input parameters for the solution matrices of the clusters  is bigger '
                              'than the actual number of input parameters')

            elif meta_model.get_in_par_means().shape[1] > model_data.shape[1] - 1:
                raise TypeError('The number of input parameters for the solution matrices of the clusters is smaller '
                                'than the actual numbers of input parameters')

            if meta_model.get_out_par_means().shape[1] != model_data.shape[2]:
                raise TypeError('The number of output parameters for the solution matrices of the clusters is not '
                                'equal to the actual numbers of input parameters')

        elif meta_model.get_type() == 'DLU':  # Additional check-up for DLU
            raise TypeError('This part is not implemented yet')

            # if not isinstance(model_data, np.ndarray):
            #     raise TypeError('The cluster input and output data is not stored in a multidimensional array')
            #
            # for clust_data in model_data:
            #
            #     if not isinstance(clust_data[0], np.matrix) or not isinstance(clust_data[1], np.matrix):
            #         raise TypeError('One of the input or output databases is not a matrix')
            #
            #     if clust_data[0].shape[1] > meta_model.get_in_par_means().shape[1]:
            #         warnings.warn('The number of input parameters for the input database of the clusters  is bigger '
            #                       'than the actual number of input parameters')
            #
            #     elif clust_data[0].shape[1] < meta_model.get_in_par_means().shape[1]:
            #         raise TypeError('The number of input parameters for the input database of the clusters is '
            #                         'smaller than the actual numbers of input parameters')
            #
            #     if clust_data[1].shape[1] > meta_model.get_out_par_means().shape[1]:
            #         raise TypeError('The number of output parameters for the output database of the clusters  is '
            #                         'bigger than the actual number of output parameters')
            #
            #     elif clust_data[1].shape[1] < meta_model.get_out_par_means().shape[1]:
            #         raise TypeError('The number of output parameters for the output database of the clusters is '
            #                         'smaller than the actual numbers of output parameters')
            #
            #     if clust_data[0].shape[0] != clust_data[1].shape[0]:
            #         raise TypeError('The number rows in the input and output database differ from each other')

        else:  # No check-up is done when the meta-model is an unknown version
            warnings.warn('The additional cluster data can not be checked, for this kind of meta-model')

        if clust_cent.shape[0] != model_data.shape[0]:
            raise TypeError('The number of clusters is different according to the number of cluster centers and'
                            'solution matrices')

        self.__clust_cent = clust_cent
        self.__model_data = model_data

    def get_clust_cent(self):
        """ A method to obtain the cluster centers

        :return: The cluster centers
        """

        return self.__clust_cent

    def get_sol_mat(self, clust):
        """ A method to obtain the solution matrix of a cluster

        :param clust: The cluster the solution matrix is needed from
        :return: The solution matrix of cluster clust
        """

        return self.__model_data[clust]

    def modify_input(self, raw_input_par):
        """ A method to prepare the raw input parameters to parameters used for calculating the output

        :param raw_input_par: A list of the raw input parameters that have to be modified
        :return: Returns the input parameter modification of the original meta-model
        """

        return self.meta_model.modify_input(raw_input_par)

    def calculate_output(self, input_par):
        """ A method to calculate the output, parameters using (modified) input parameters

        :param input_par: A list of the input parameters that are used to calculate the output
        :return: Raises an error since this is an abstract class, subclasses have to implement this method
        """

        raise NotImplementedError


class ClosestClusterDecorator(ClusterDecorator):
    """ A decorator class for different meta-model additions based on clustering, this one particular on closest
    clustering

    Attributes:
        mm_type: The type of the meta-model decoration (In this Decorated Cluster class, it is 'Decorated Cluster ')
        meta_model: The meta_model that is decorated
        clust_cent: The locations of the cluster centers per input parameter
        model_data: A 3-dimensional array with the solution matrices for all different clusters
    """

    meta_model = None
    mm_type = 'Closest Cluster '

    def calculate_output(self, input_par):
        """ A method to calculate the output, parameters using (modified) input parameters

        :param input_par: A list of the input parameters that are used to calculate the output
        :return: Gives the correct output for this particular input
        """

        if input_par.shape[1] != self.get_clust_cent().shape[1]:
            raise TypeError('The input parameters (%x) and the input database (%x) do not have a matching size' %
                            (input_par.shape[0], self.get_clust_cent().shape[1]))

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

        # Initialization of the closest cluster center index at the start
        closest_comb_index = find_euc_dist(self.get_clust_cent()[0], input_par)
        index = 0

        # The cluster center is gone through to find the combination with the lowest Euclidian distance
        for i in range(self.get_clust_cent().shape[0]):
            new_comb_index = find_euc_dist(self.get_clust_cent()[i], input_par)
            if closest_comb_index > new_comb_index:
                closest_comb_index = new_comb_index
                index = i

        # Find the correct way to output the data

        sol_mat = self.get_sol_mat(index)
        regress_coeff = sol_mat[1:]
        output_const = sol_mat[0]

        # Return the output data corresponding to the closest cluster center index
        output_var = input_par * regress_coeff
        output_par = np.add(output_const, output_var)

        # Return the output data corresponding to the closest cluster center index
        return output_par

#
# class MultipleClusterDecorator(ClusterDecorator):
#     """ A decorator class for different meta-model additions based on clustering, this one particular on using multu
#         clustering
#
#         Attributes:
#             mm_type: The type of the meta-model decoration (In this Decorated Cluster class, it is 'Decorated Cluster ')
#             meta_model: The meta_model that is decorated
#             clust_cent: The locations of the cluster centers per input parameter
#             model_data: A 3-dimensional array with the solution matrices for all different clusters
#         """
#
#     meta_model = None
#     mm_type = 'Closest Cluster '
#
#     def calculate_output(self, input_par):
#         """ A method to calculate the output, parameters using (modified) input parameters
#
#         :param input_par: A list of the input parameters that are used to calculate the output
#         :return: Gives the correct output for this particular input
#         """
#
#         if input_par.shape[1] != self.__clust_cent.shape[1]:
#             raise TypeError('The input parameters (%x) and the input database (%x) do not have a matching size' %
#                             (input_par.shape[0], self.__clust_cent.shape[1]))
#
#         def find_euc_dist(list_par1, list_par2):
#             """ Finds the Euclidian distance between the two parameter points
#
#             :param list_par1: List with the parameter values of point 1
#             :param list_par2: List with the parameter values of point 2
#             :return: The euclidian distance between the two values
#             """
#
#             # First find the squared Euclidian distance
#             temp_euc_dist = np.subtract(list_par1, list_par2)
#             euc_dist = np.sum(np.square(temp_euc_dist))
#             return pow(euc_dist, .5)
#
#         # Initialization of the closest cluster center index at the start
#         closest_comb_index = find_euc_dist(self.__clust_cent[1], input_par)
#         index = 0
#
#         # The cluster center is gone through to find the combination with the lowest Euclidian distance
#         for i in range(self.__clust_cent.shape[0]):
#             new_comb_index = find_euc_dist(self.__clust_cent[i], input_par)
#
#             if closest_comb_index > new_comb_index:
#                 closest_comb_index = new_comb_index
#                 index = i
#
#         # Find the correct way to output the data
#         sol_mat = self.__model_data[index]
#         regress_coeff = sol_mat[1:]
#         output_const = sol_mat[0]
#
#         # Return the output data corresponding to the closest cluster center index
#         output_var = input_par * regress_coeff
#         output_par = np.add(output_const, output_var)
#
#         return output_par
