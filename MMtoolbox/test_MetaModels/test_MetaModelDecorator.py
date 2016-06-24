from unittest import TestCase
from MetaModels import MetaModel as MM
from MetaModels import MetaModelDecorator as MMD
from test_MetaModels import test_MetaModel as tMM
import numpy as np
import warnings


class TestModelDecorator(tMM.TestAbstractModel):
    """ All test for the modelDecorator to see if this decorator works as it should

    """

    # Construct an example PLSR meta-model
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    def construct_model(self):
        """ Constructs a meta-model for testing

        :return: A meta-model
        """

        meta_model_base = MM.AbstractModel(self.__in_par_intervals, self.__in_par_means, self.__in_par_variances,
                                                  self.__out_par_intervals, self.__out_par_means, self.__out_par_variances)

        return MMD.ModelDecorator(meta_model_base)

    def test_initialization(self):
        """ Tests if the initialization is done properly

        :return: A result for the initialization testing
        """

        self.assertRaises(TypeError, MMD.ModelDecorator, 'not a meta-model')

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """

        test_model = self.construct_model()

        self.failUnlessEqual(test_model.get_type(), 'Decorated Abstract Abstract')


class TestInputDecorator(TestModelDecorator):
        """ All test for the Input Decorator to see if this decorator works as it should

        """

        # Construct an example PLSR meta-model
        __sol_mat = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001]])
        __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
        __in_par_means = np.mat([1.2, 0.3])
        __in_par_variances = np.mat([0.1, 0.001])
        __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
        __out_par_means = np.mat([2.2, 1.3])
        __out_par_variances = np.mat([0.1, 0.001])

        def construct_model(self):
            """ Constructs a meta-model for testing

                :return: A meta-model
            """

            meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                      self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                                      self.__out_par_variances)

            return MMD.InputDecorator(meta_model_base)

        def test_get_type(self):
            """ Tests whether the function AbstractModel.get_type returns the right type

            :return: The meta-model type tested
            """

            test_model = self.construct_model()

            np.testing.assert_array_equal(test_model.get_type(), 'Decorated Input PLSR')

        def test_calculate_output(self):
            """ Tests if the calculated output is the correct for this decorator

            """

            test_model = self.construct_model()

            # Calculate the right output first and then compare it with the output from the function
            input_par = np.mat([[1.1, 0.4]])
            output_par = np.zeros((1, 2))

            # First output value
            output_par[0, 0] = self.__sol_mat[0, 0]
            output_par[0, 0] += self.__sol_mat[1, 0] * input_par[0, 0] + self.__sol_mat[2, 0] * input_par[0, 1]

            # Second output value
            output_par[0, 1] = self.__sol_mat[0, 1]
            output_par[0, 1] += self.__sol_mat[1, 1] * input_par[0, 0] + self.__sol_mat[2, 1] * input_par[0, 1]

            np.testing.assert_array_equal(test_model.calculate_output(input_par), output_par)

        def test_get_input_spec(self):
            """ Tests if the input specifications are past through normally

            """

            test_model = self.construct_model()

            self.failUnlessEqual(test_model.get_input_spec(), 'Abstract')


class TestPolynomialInputDecorator(TestInputDecorator):
    """ All test for the Polynomial Decorator to see if this decorator works as it should

    """

    # Construct an example PLSR meta-model
    __sol_mat = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001], [1, 1], [0.1, 0.01], [0.1, 0.001]])
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    def construct_model(self):
        """ Constructs a meta-model for testing

            :return: A meta-model
        """

        meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                                  self.__out_par_variances)

        return MMD.PolynomialInputDecorator(meta_model_base)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """

        test_model = self.construct_model()

        self.failUnlessEqual(test_model.get_type(), 'Polynomial Input PLSR')

    def test_standardize_input(self):
        """ Tests if the input is correctly standardized

        :return: A result of the test if it is correctly standardized
        """

        test_model = self.construct_model()

        # The raw and modified input parameters
        input_par_1 = np.mat([[1.1, 0.4]])

        mean_par_1 = np.subtract(input_par_1, self.__in_par_means)
        mod_par_1 = np.divide(mean_par_1, np.sqrt(self.__in_par_variances))

        np.testing.assert_array_equal(test_model.standardize_input(input_par_1), mod_par_1)

    def test_modify_input(self):
        """ Tests if the input is correctly modified

        :return: A result of the test if it is correctly modified
        """

        test_model = self.construct_model()

        # The raw and modified input parameters
        input_par_1 = np.mat([[1.1, 0.4]])

        nr_par = int(input_par_1.shape[1])
        mod_input_par_1 = np.mat(np.zeros((pow(nr_par, 2) + 3 * nr_par) / 2))
        stand_par = test_model.standardize_input(input_par_1)
        mod_input_par_1[0, range(nr_par)] = stand_par[0]

        # Add all terms of the polynomial input parameters to the modified input parameters
        for i in range(nr_par):
            for j in range(i, nr_par):
                mod_input_par_1[0, ((nr_par - 1) * (i + 1) + j + 1)] = mod_input_par_1[0, i] * mod_input_par_1[0, j]

        np.testing.assert_array_equal(test_model.modify_input(input_par_1), mod_input_par_1)

    def test_calculate_output(self):
        """ Tests if the calculated output is the correct for this decorator

        """

        test_model = self.construct_model()

        # Calculate the right output first and then compare it with the output from the function
        input_par = np.mat([[1.1, 0.4]])
        mod_input_par = test_model.modify_input(input_par)

        var_par = mod_input_par * self.__sol_mat[1:]
        output_par = np.add(self.__sol_mat[0], var_par)

        np.testing.assert_array_equal(test_model.calculate_output(mod_input_par), output_par)

    def test_simulate(self):
        """ Tests if the calculated output is the correct for this decorator

        """

        test_model = self.construct_model()

        input_par = np.mat([[1.1, 0.4]])
        mod_input_par = test_model.modify_input(input_par)
        output_par = test_model.calculate_output(mod_input_par)

        np.testing.assert_array_equal(test_model.simulate(input_par), output_par)

    def test_get_input_spec(self):
        """ Tests if the input specifications are past through normally

        """

        test_model = self.construct_model()
        self.failUnlessEqual(test_model.get_input_spec(), 'Polynomial')


class TestModifiedInputDecorator(TestInputDecorator):
    """ All test for the Polynomial Decorator to see if this decorator works as it should

    """

    # Construct an example PLSR meta-model
    __sol_mat = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001], [1, 1], [0.1, 0.01], [0.1, 0.001], [1, 1], [0.1, 0.01],
                        [0.1, 0.001], [1, 1], [0.1, 0.01], [0.1, 0.001], [1, 1], [0.1, 0.01], [0.1, 0.001], [1, 1],
                        [0.1, 0.01], [0.1, 0.001], [1, 1]])
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])
    __modifiers = ['log', 'sqr', 'root', 'inv', 'exp', 'sin', 'cos', 'tan']

    def construct_model(self):
        """ Constructs a meta-model for testing

            :return: A meta-model
        """

        meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                                  self.__out_par_variances)

        return MMD.ModifiedInputDecorator(meta_model_base, self.__modifiers)

    def test_initialization(self):
        """ Tests if this Decorator is properly initialized

        :return:
        """

        modifiers_1 = [1] # no strings
        modifiers_2 = ['not one of the types']

        meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                                  self.__out_par_variances)

        TestInputDecorator.test_initialization(self)

        self.assertRaises(TypeError, MMD.ModifiedInputDecorator, meta_model_base, modifiers_1)
        self.assertRaises(ValueError, MMD.ModifiedInputDecorator, meta_model_base, modifiers_2)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """

        test_model = self.construct_model()

        self.failUnlessEqual(test_model.get_type(), 'Modified Input PLSR')

    def test_modify_input(self):
        """ Tests if the input is correctly modified

        :return: A result of the test if it is correctly modified
        """

        test_model = self.construct_model()

        # The raw and modified input parameters
        input_par_1 = np.mat([[1.1, 0.4]])

        nr_par = int(input_par_1.shape[1])
        mod_input_par_1 = np.mat(np.zeros((len(self.__modifiers) + 1) * nr_par))
        mod_input_par_1[0, range(nr_par)] = input_par_1[0]

        for i in range(len(self.__modifiers)):
            if self.__modifiers[i] == 'log':
                temp_par = np.log(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif self.__modifiers[i] == 'sqr':
                temp_par = np.square(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par
            elif self.__modifiers[i] == 'root':
                temp_par = np.sqrt(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par
            elif self.__modifiers[i] == 'inv':
                temp_par = np.power(input_par_1, -1.)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif self.__modifiers[i] == 'exp':
                temp_par = np.exp(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif self.__modifiers[i] == 'sin':
                temp_par = np.sin(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif self.__modifiers[i] == 'cos':
                temp_par = np.cos(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif self.__modifiers[i] == 'tan':
                temp_par = np.tan(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]

        np.testing.assert_array_equal(test_model.modify_input(input_par_1), mod_input_par_1)

    def test_calculate_output(self):
        """ Tests if the calculated output is the correct for this decorator

        """

        test_model = self.construct_model()

        # Calculate the right output first and then compare it with the output from the function
        input_par = np.mat([[1.1, 0.4]])
        mod_input_par = test_model.modify_input(input_par)

        print(mod_input_par.shape)

        var_par = mod_input_par * self.__sol_mat[1:]
        output_par = np.add(self.__sol_mat[0], var_par)

        np.testing.assert_array_equal(test_model.calculate_output(mod_input_par), output_par)

    def test_simulate(self):
        """ Tests if the calculated output is the correct for this decorator

        """

        test_model = self.construct_model()

        input_par = np.mat([[1.1, 0.4]])
        mod_input_par = test_model.modify_input(input_par)
        output_par = test_model.calculate_output(mod_input_par)

        np.testing.assert_array_equal(test_model.simulate(input_par), output_par)

    def test_get_input_spec(self):
        """ Tests if the input specifications are past through normally

        """

        test_model = self.construct_model()
        self.failUnlessEqual(test_model.get_input_spec(), self.__modifiers)


class TestClusterDecorator(TestModelDecorator):
    """ All test for the Input Decorator to see if this decorator works as it should

    """

    # Construct an example PLSR meta-model
    __sol_mat = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001]])

    # Construct an example DLU meta-model
    __input_data = np.mat([[0.75, 0.25], [0.75, 0.35], [1.25, 0.25], [1.25, 0.35]])
    __output_data = np.mat([[1.75, 1.25], [1.75, 1.35], [2.25, 1.25], [2.25, 1.35]])

    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    __sol_mat_clust = np.array([[[1, 1], [0.1, 0.01], [0.1, 0.001]], [[0.9, 0.9], [0.09, 0.009], [0.09, 0.0009]]])
    __clust_cent = np.mat([[0.75, 0.25], [1.25, 0.35]])

    def construct_model(self):
        """ Constructs a meta-model for testing

            :return: A meta-model
        """

        meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                                  self.__out_par_variances)

        return MMD.ClusterDecorator(meta_model_base, self.__clust_cent, self.__sol_mat_clust)

    def test_initialization(self):
        """ Tests if the initialization works properly

        :return: A result of the initialization test
        """

        def test_init(clust_cent, sol_mat_clust, error):
            """ Tests if the initialization is done properly with the cluster centers and the solution matrices for
            the clusters

            :param clust_cent: The cluster centers
            :param sol_mat_clust: The solution matrices for the clusters
            :param error: The type of errror that should occur
            :return: The initialization errors testes
            """

            meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                      self.__in_par_variances, self.__out_par_intervals,
                                                      self.__out_par_means,
                                                      self.__out_par_variances)

            self.assertRaises(error, MMD.ClusterDecorator, meta_model_base, clust_cent, sol_mat_clust)

        # First the cluster centers

        # Check if the cluster centers are of the correct type
        clust_cent_1 = 'not a matrix'
        clust_cent_2 = np.matrix([[1, 2.], ['not a number', .5]])

        test_init(clust_cent_1, self.__sol_mat_clust, TypeError)
        test_init(clust_cent_2, self.__sol_mat_clust, TypeError)

        # Check if the cluster centers are of the correct size
        clust_cent_3 = np.mat([[0.75, 0.25, 1.2], [1.25, 0.35, 3.2]])
        clust_cent_4 = np.mat([[0.75], [1.25]])

        warnings.simplefilter("error")

        test_init(clust_cent_3, self.__sol_mat_clust, UserWarning)
        test_init(clust_cent_4, self.__sol_mat_clust, TypeError)

        warnings.simplefilter("ignore")

        # Check if the cluster centers are within the predefined intervals
        clust_cent_5 = np.mat([[1.75, 0.25], [1.25, 0.35]])
        clust_cent_6 = np.mat([[0.75, 0.25], [1.25, 0.15]])

        test_init(clust_cent_5, self.__sol_mat_clust, ValueError)
        test_init(clust_cent_6, self.__sol_mat_clust, ValueError)

        # Then the solution matrices for PLSR

        # Check if the cluster solution matrices are of the correct type
        sol_mat_clust_1 = 'not an array'
        sol_mat_clust_2 = np.array([[2, 1], [1, 2]])  # Not 3 dimensional
        sol_mat_clust_3 = np.array([[[1, 1], [0.1, 0.01], [0.1, 0.001]], [[0.9, 0.9], ['not a number', 0.009],
                                                                          [0.09, 0.0009]]])

        test_init(self.__clust_cent, sol_mat_clust_1, TypeError)
        test_init(self.__clust_cent, sol_mat_clust_2, TypeError)
        test_init(self.__clust_cent, sol_mat_clust_3, TypeError)

        # Check if the cluster solution matrices are of the correct size
        sol_mat_clust_4 = np.array([[[1, 1], [0.1, 0.01], [0.1, 0.001], [1, 1]], [[0.9, 0.9], [0.09, 0.009],
                                                                          [0.09, 0.0009], [1, 1]]])
        sol_mat_clust_5 = np.array([[[1, 1], [0.1, 0.01]], [[0.9, 0.9], [0.09, 0.009]]])
        sol_mat_clust_6 = np.array([[[1, 1, 1], [0.1, 0.01, 1], [0.1, 0.001, 1]], [[0.9, 0.9, 1], [0.09, 0.009, 1],
                                                                                   [0.09, 0.0009, 1]]])
        sol_mat_clust_7 = np.array([[[1], [0.1], [0.1]], [[0.9], [0.09], [0.09]]])

        warnings.simplefilter("error")

        test_init(self.__clust_cent, sol_mat_clust_4, UserWarning)
        test_init(self.__clust_cent, sol_mat_clust_5, TypeError)
        test_init(self.__clust_cent, sol_mat_clust_6, TypeError)
        test_init(self.__clust_cent, sol_mat_clust_7, TypeError)

        warnings.simplefilter("ignore")

        # Check if the number of clusters is the same for the cluster centers and the solution matrices
        sol_mat_clust_8 = np.array([[[1, 1], [0.1, 0.01], [0.1, 0.001]], [[0.9, 0.9], [0.09, 0.009], [0.09, 0.0009]],
                                    [[0.9, 0.9], [0.09, 0.009], [0.09, 0.0009]]])
        clust_cent_7 = np.mat([[0.75, 0.25], [1.25, 0.35], [1.25, 0.35]])

        test_init(self.__clust_cent, sol_mat_clust_8, TypeError)
        test_init(clust_cent_7, self.__sol_mat_clust, TypeError)


        # # Then the input and output database
        # database_1 = 'not an n-dimensional array'
        # database_2 = np.array([[self.__output_data, 'not a matrix'], [self.__input_data_2, self.__output_data_2]])
        #
        # input_data_3 = np.mat([[0.75, 0.25, 0.1], [0.75, 0.35, 0.1], [1.25, 0.25, 0.1], [1.25, 0.35, 0.1]])
        # # database_3 = np.array([[input_data_3, self.__output_data], [self.__input_data_2, self.__output_data_2]])
        #
        # input_data_4 = np.mat([[0.75], [0.75], [1.25], [1.25]])
        # database_4 = np.array([[input_data_4, self.__output_data], [self.__input_data_2, self.__output_data_2]])
        #
        # output_data_5 = np.mat([[0.75, 0.25, 0.1], [0.75, 0.35, 0.1], [1.25, 0.25, 0.1], [1.25, 0.35, 0.1]])
        # database_5 = np.array([[self.__input_data, output_data_5], [self.__input_data_2, self.__output_data_2]])
        #
        # output_data_6 = np.mat([[0.75], [0.75], [1.25], [1.25]])
        # database_6 = np.array([[self.__input_data, output_data_6], [self.__input_data_2, self.__output_data_2]])
        #
        # input_data_7 = np.mat([[0.75, 0.25], [0.75, 0.35], [1.25, 0.25]])
        # database_7 = np.array([[input_data_7, self.__output_data], [self.__input_data_2, self.__output_data_2]])
        #
        # input_data_8 = np.mat([[0.75, 0.25], [0.75, 0.35], [1.25, 0.25], [0.75, 0.35], [1.25, 0.25]])
        # database_8 = np.array([[input_data_8, self.__output_data], [self.__input_data_2, self.__output_data_2]])
        #
        # test_init_DLU(self.__clust_cent, database_1, TypeError)
        # test_init_DLU(self.__clust_cent, database_2, TypeError)
        #
        # warnings.simplefilter("error")
        # test_init_DLU(self.__clust_cent, database_3, UserWarning)
        # warnings.simplefilter("ignore")
        #
        # test_init_DLU(self.__clust_cent, database_4, TypeError)
        # test_init_DLU(self.__clust_cent, database_5, TypeError)
        # test_init_DLU(self.__cl ust_cent, database_6, TypeError)
        # test_init_DLU(self.__clust_cent, database_7, TypeError)
        # test_init_DLU(self.__clust_cent, database_8, TypeError)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """

        test_model = self.construct_model()

        np.testing.assert_array_equal(test_model.get_type(), 'Decorated Cluster PLSR')

    def test_get_clust_cent(self):
        """ Tests if the method to get the cluster centers works properly

        :return: The result of testing the get_clust_cent method
        """

        test_model = self.construct_model()

        np.testing.assert_array_equal(test_model.get_clust_cent(), self.__clust_cent)

    def test_get_sol_mat(self):
        """ Tests if hte method to get the solution matrix for different clusters works properly

        :return:  The result of testing the get_sol_mat method
        """

        test_model = self.construct_model()

        for i in range(self.__clust_cent.shape[0]):
            np.testing.assert_array_equal(test_model.get_sol_mat(i), self.__sol_mat_clust[i])

    def test_modify_input(self):
        """ Tests if the modify_input method works properly

        :return: The result of testing the modify_input method
        """

        test_model = self.construct_model()

        input_par = np.mat([1.1, 0.4])

        mod_input_par = test_model.meta_model.modify_input(input_par)

        np.testing.assert_array_equal(test_model.modify_input(input_par), mod_input_par)


class TestClosestClusterDecorator(TestClusterDecorator):
    """ All tests for the closest cluster decorator, whether it is correct

    """

    # Construct an example PLSR meta-model
    __sol_mat = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001]])

    # Construct an example DLU meta-model
    __input_data = np.mat([[0.75, 0.25], [0.75, 0.35], [1.25, 0.25], [1.25, 0.35]])
    __output_data = np.mat([[1.75, 1.25], [1.75, 1.35], [2.25, 1.25], [2.25, 1.35]])

    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    __sol_mat_clust = np.array([[[1, 1], [0.1, 0.01], [0.1, 0.001]], [[0.9, 0.9], [0.09, 0.009], [0.09, 0.0009]]])
    __clust_cent = np.mat([[0.75, 0.25], [1.25, 0.35]])

    def construct_model(self):
        """ Constructs a meta-model for testing

            :return: A meta-model
        """

        meta_model_base = MM.PLSRMetaModel(self.__sol_mat, self.__in_par_intervals, self.__in_par_means,
                                                  self.__in_par_variances, self.__out_par_intervals,
                                                  self.__out_par_means,
                                                  self.__out_par_variances)

        return MMD.ClosestClusterDecorator(meta_model_base, self.__clust_cent, self.__sol_mat_clust)

    def test_get_type(self):
        """ Tests whether the function ClosestClusterDecorator.get_type returns the right type

        :return: The meta-model type tested
        """

        test_model = self.construct_model()

        np.testing.assert_array_equal(test_model.get_type(), 'Closest Cluster PLSR')

    def test_calculate_output(self):
        """ Tests if the output is calculated correctly with clusters

        :return: The calculated output tested
        """

        test_model = self.construct_model()

        # test to find if the regression coefficient size matches the number of input parameters
        input_par_1 = np.mat([[1.1, 0.4, 1.2]])
        input_par_2 = np.mat([[1.1]])

        self.assertRaises(TypeError, test_model.calculate_output, input_par_1)
        self.assertRaises(TypeError, test_model.calculate_output, input_par_2)

        # Calculate the right output first and then compare it with the output from the function
        input_par = np.mat([[0.65, 0.20]])
        output_par = np.zeros((1, 2))

        # First output value
        output_par[0, 0] = self.__sol_mat_clust[0, 0, 0]
        output_par[0, 0] += self.__sol_mat_clust[0, 1, 0] * input_par[0,0] + self.__sol_mat_clust[0, 2, 0] * input_par[0,1]

        # Second output value
        output_par[0, 1] = self.__sol_mat_clust[0, 0, 1]
        output_par[0, 1] += self.__sol_mat_clust[0, 1, 1] * input_par[0, 0] + self.__sol_mat_clust[0, 2, 1] * input_par[0, 1]

        np.testing.assert_array_equal(test_model.calculate_output(input_par), output_par)

        # Calculate the right output first and then compare it with the output from the function
        input_par_2 = np.mat([[1.05, 0.4]])
        output_par_2 = np.zeros((1, 2))

        # First output value
        output_par_2[0, 0] = self.__sol_mat_clust[1, 0, 0]
        output_par_2[0, 0] += self.__sol_mat_clust[1, 1, 0] * input_par_2[0, 0] + self.__sol_mat_clust[1, 2, 0] * \
                                                                                input_par_2[0, 1]

        # Second output value
        output_par_2[0, 1] = self.__sol_mat_clust[1, 0, 1]
        output_par_2[0, 1] += self.__sol_mat_clust[1, 1, 1] * input_par_2[0, 0] + self.__sol_mat_clust[1, 2, 1] * \
                                                                                input_par_2[0, 1]

        print(test_model.calculate_output(input_par_2))

        np.testing.assert_array_equal(test_model.calculate_output(input_par_2), output_par_2)