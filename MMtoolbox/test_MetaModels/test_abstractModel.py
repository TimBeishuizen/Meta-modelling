from unittest import TestCase
from MetaModels import Models
import numpy as np
import warnings


class TestAbstractModel(TestCase):
    """ A class to test the AbstractModel meta-model structure

    """

    # Construct an example Abstract meta-model
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    test_model = Models.AbstractModel(__in_par_intervals, __in_par_means, __in_par_variances, __out_par_intervals,
                                        __out_par_means, __out_par_variances)

    def test_initialization(self):
        """ Tests if the initialization is done correctly. It tests all input for the initialization
        with slight changes to check if they work

        :return: A confirmation if the test work or not.
        """

        def construct_abstract_model(changed_value, changed_value_place, type_error):
            """ A frame to test the different kinds of errors that can occur for the initialization of a meta-model

            :param changed_value: The value for the meta-model to be changed
            :param changed_value_place: The location for the meta-model to be changed - 1 = in_par_intervals,
                    2 = in_par_means, 3 = in_par_variances, 4 = out_par_intervals, 5 = out_par_means,
                    6 = out_par_variances
            :param type_error: The type of error that should have occurred.
            """

            if changed_value_place == 1:
                self.assertRaises(type_error, Models.AbstractModel, changed_value, self.__in_par_means,
                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                  self.__out_par_variances)
            elif changed_value_place == 2:
                self.assertRaises(type_error, Models.AbstractModel, self.__in_par_intervals, changed_value,
                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                  self.__out_par_variances)
            elif changed_value_place == 3:
                self.assertRaises(type_error, Models.AbstractModel, self.__in_par_intervals, self.__in_par_means,
                                  changed_value, self.__out_par_intervals, self.__out_par_means,
                                  self.__out_par_variances)
            elif changed_value_place == 4:
                self.assertRaises(type_error, Models.AbstractModel, self.__in_par_intervals, self.__in_par_means,
                                  self.__in_par_variances, changed_value, self.__out_par_means,
                                  self.__out_par_variances)
            elif changed_value_place == 5:
                self.assertRaises(type_error, Models.AbstractModel, self.__in_par_intervals, self.__in_par_means,
                                  self.__in_par_variances, self.__out_par_intervals, changed_value,
                                  self.__out_par_variances)
            elif changed_value_place == 6:
                self.assertRaises(type_error, Models.AbstractModel, self.__in_par_intervals, self.__in_par_means,
                                  self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                                  changed_value)

        # Control test, to check if a normal case would work
        try:
            Models.AbstractModel(self.__in_par_intervals, self.__in_par_means, self.__in_par_variances,
                                 self.__out_par_intervals, self.__out_par_means, self.__out_par_variances)
        except ValueError:
            self.fail('Initialization threw an input value error')

        # Test the input parameter intervals construction
        new_in_par_intervals_1 = 'not a list'
        new_in_par_intervals_2 = np.mat([[0.5, 1.5], 'not a nested list'])
        new_in_par_intervals_3a = np.mat([[0.5, 1.5], [0.2, 0.4, 0.5]])  # Too many floats for input parameters
        new_in_par_intervals_3b = np.mat([[0.5, 1.5], [0.5]])             # Too few floats for input parameters
        new_in_par_intervals_4 = np.mat([[0.5, 1.5], ['no', 'floats']])

        construct_abstract_model(new_in_par_intervals_1, 1, TypeError)
        construct_abstract_model(new_in_par_intervals_2, 1, TypeError)
        construct_abstract_model(new_in_par_intervals_3a, 1, TypeError)
        construct_abstract_model(new_in_par_intervals_3b, 1, TypeError)
        construct_abstract_model(new_in_par_intervals_4, 1, TypeError)

        # Test the input parameter means construction
        new_in_par_means_1 = 'not a list'
        new_in_par_means_2 = np.mat([1.2, 'no float'])

        construct_abstract_model(new_in_par_means_1, 2, TypeError)
        construct_abstract_model(new_in_par_means_2, 2, TypeError)

        # Test the input parameter variances construction
        new_in_par_variances_1 = 'not a list'
        new_in_par_variances_2 = np.mat([0.1, 'no float'])

        construct_abstract_model(new_in_par_variances_1, 3, TypeError)
        construct_abstract_model(new_in_par_variances_2, 3, TypeError)

        # Test the output parameter intervals construction
        new_out_par_intervals_1 = 'not a list'
        new_out_par_intervals_2 = np.mat([[1.5, 2.5], 'not a nested list'])
        new_out_par_intervals_3a = np.mat([[1.5, 2.5], [1.2, 1.3, 1.4]])   # Too many floats for output parameters
        new_out_par_intervals_3b = np.mat([[1.5, 2.5], [1.2]])          # Too few floats for output parameters
        new_out_par_intervals_4 = np.mat([[1.5, 2.5], ['no', 'floats']])

        construct_abstract_model(new_out_par_intervals_1, 4, TypeError)
        construct_abstract_model(new_out_par_intervals_2, 4, TypeError)
        construct_abstract_model(new_out_par_intervals_3a, 4, TypeError)
        construct_abstract_model(new_out_par_intervals_3b, 4, TypeError)
        construct_abstract_model(new_out_par_intervals_4, 4, TypeError)

        # Test the output parameter means construction
        new_out_par_means_1 = 'not a list'
        new_out_par_means_2 = np.mat([1.2, 'no float'])

        construct_abstract_model(new_out_par_means_1, 5, TypeError)
        construct_abstract_model(new_out_par_means_2, 5, TypeError)

        # Test the output parameter variances  construction
        new_out_par_variances_1 = 'not a list'
        new_out_par_variances_2 = np.mat([0.1, 'no float'])

        construct_abstract_model(new_out_par_variances_1, 6, TypeError)
        construct_abstract_model(new_out_par_variances_2, 6, TypeError)

        # Test for different number of input parameters
        new_in_par_intervals_5a = np.mat([[0.5, 1.5], [0.2, 0.4], [1, 2]])
        new_in_par_intervals_5b = np.mat([[0.5, 1.5]])
        new_in_par_means_3a = np.mat([1.2, 0.3, 1.4])
        new_in_par_means_3b = np.mat([1.2])
        new_in_par_variances_3a = np.mat([0.1, 0.001, 0.1])
        new_in_par_variances_3b = np.mat([0.1])

        construct_abstract_model(new_in_par_intervals_5a, 1, TypeError)
        construct_abstract_model(new_in_par_intervals_5b, 1, TypeError)
        construct_abstract_model(new_in_par_means_3a, 2, TypeError)
        construct_abstract_model(new_in_par_means_3b, 2, TypeError)
        construct_abstract_model(new_in_par_variances_3a, 3, TypeError)
        construct_abstract_model(new_in_par_variances_3b, 3, TypeError)

        # Test for different number of output parameters
        new_out_par_intervals_5a = np.mat([[1.5, 2.5], [1.2, 1.4], [2, 3]])
        new_out_par_intervals_5b = np.mat([[1.5, 2.5]])
        new_out_par_means_3a = np.mat([2.2, 1.3, 2.4])
        new_out_par_means_3b = np.mat([2.2])
        new_out_par_variances_3a = np.mat([0.1, 0.001, 0.1])
        new_out_par_variances_3b = np.mat([0.1])

        construct_abstract_model(new_out_par_intervals_5a, 4, TypeError)
        construct_abstract_model(new_out_par_intervals_5b, 4, TypeError)
        construct_abstract_model(new_out_par_means_3a, 5, TypeError)
        construct_abstract_model(new_out_par_means_3b, 5, TypeError)
        construct_abstract_model(new_out_par_variances_3a, 6, TypeError)
        construct_abstract_model(new_out_par_variances_3b, 6, TypeError)

        # Test if the intervals are formulated correctly, i.e. lower boundary first, upper boundary second
        new_in_par_intervals_6 = np.mat([[1.5, 0.5], [0.2, 0.4]])
        new_out_par_intervals_6 = np.mat([[1.5, 2.5], [1.4, 1.2]])

        construct_abstract_model(new_in_par_intervals_6, 1, ValueError)
        construct_abstract_model(new_out_par_intervals_6, 4, ValueError)

        # Test if the input mean and variance values are between the intervals
        new_in_par_means_4a = np.mat([1.6, 0.3])
        new_in_par_means_4b = np.mat([1.2, 0.1])
        new_in_par_variances_4 = np.mat([10, 1.0])

        construct_abstract_model(new_in_par_means_4a, 2, ValueError)
        construct_abstract_model(new_in_par_means_4b, 2, ValueError)
        construct_abstract_model(new_in_par_variances_4, 3, ValueError)

        # Test if the output mean and variance values are between the intervals
        new_out_par_means_4a = np.mat([2.6, 1.3])
        new_out_par_means_4b = np.mat([2.2, 1.1])
        new_out_par_variances_4 = np.mat([10, 0.0001])

        construct_abstract_model(new_out_par_means_4a, 5, ValueError)
        construct_abstract_model(new_out_par_means_4b, 5, ValueError)
        construct_abstract_model(new_out_par_variances_4, 6, ValueError)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """
        self.failUnlessEqual(self.test_model.get_type(), 'Abstract')

    def test_get_in_par_intervals(self):
        """ Tests whether the function AbstractModel.get_in_par_intervals returns the right values

        :return: The meta-model input parameter intervals tested
        """
        self.failUnlessEqual(self.test_model.get_in_par_intervals().all(), self.__in_par_intervals.all())

    def test_get_in_par_means(self):
        """ Tests whether the function AbstractModel.get_in_par_means returns the right values

        :return: The meta-model input parameter means tested
        """
        self.failUnlessEqual(self.test_model.get_in_par_means().all(), self.__in_par_means.all())

    def test_get_in_par_variances(self):
        """ Tests whether the function AbstractModel.get_in_par_variances returns the right values

        :return: The meta-model input parameter variances tested
        """
        self.failUnlessEqual(self.test_model.get_in_par_variances().all(), self.__in_par_variances.all())

    def test_get_out_par_intervals(self):
        """ Tests whether the function AbstractModel.get_out_par_intervals returns the right values

        :return: The meta-model output parameter intervals tested
        """
        self.failUnlessEqual(self.test_model.get_out_par_intervals().all(), self.__out_par_intervals.all())

    def test_get_out_par_means(self):
        """ Tests whether the function AbstractModel.get_out_par_means returns the right values

        :return: The meta-model output parameter means tested
        """
        self.failUnlessEqual(self.test_model.get_out_par_means().all(), self.__out_par_means.all())

    def test_get_out_par_variances(self):
        """ Tests whether the function AbstractModel.get_out_par_variances returns the right values

        :return: The meta-model output parameter variances tested
        """
        self.failUnlessEqual(self.test_model.get_out_par_variances().all(), self.__out_par_variances.all())

    def test_simulate(self):
        """ Tests whether input of AbstractMode.simulate has the correct input.

        :return: The simulated parameter output tested
        """

        # Test if the number of input parameters is the same as for the meta-model
        raw_input_par_1a = np.mat([1.4, 0.25, 0.1])
        raw_input_par_1b = np.mat([1.4])

        self.assertRaises(TypeError, self.test_model.simulate, raw_input_par_1a)
        self.assertRaises(TypeError, self.test_model.simulate, raw_input_par_1b)

        # Test if the input parameters are defined correctly
        raw_input_par_2a = 'Not a list'
        raw_input_par_2b = np.mat([1.4, 'Not a float'])

        self.assertRaises(TypeError, self.test_model.simulate, raw_input_par_2a)
        self.assertRaises(TypeError, self.test_model.simulate, raw_input_par_2b)

        # Test if the input parameters are in between the intervals and gives a warning if not
        raw_input_par_3a = np.mat([0.4, 0.25])
        raw_input_par_3b = np.mat([1.4, 0.45])

        warnings.simplefilter("error")

        self.assertRaises(UserWarning, self.test_model.simulate, raw_input_par_3a)
        self.assertRaises(UserWarning, self.test_model.simulate, raw_input_par_3b)

        warnings.simplefilter("ignore")


class TestPLSRMetaModel(TestAbstractModel):
    """ A class to test the PLSR meta-model structure

    """

    # Construct an example PLSR meta-model
    __sol_mat = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001]])
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    test_model = Models.PLSRMetaModel(__sol_mat, __in_par_intervals, __in_par_means, __in_par_variances,
                                        __out_par_intervals, __out_par_means, __out_par_variances)

    def test_initialization(self):
        """ A test for the initialization of the PLSRMetaModel

        :return: A PLSR meta-model tested
        """

        def test_sol_mat(sol_mat):
            self.assertRaises(TypeError, Models.AbstractModel, sol_mat, self.__in_par_means, self.__in_par_means,
                              self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                              self.__out_par_variances)

        # Check if not the proper input for the sol mat is used
        sol_mat_1 = 'not a list'
        sol_mat_2 = np.mat([[1, 1], [0.1, 0.01], 'not a list'])
        sol_mat_3 = np.mat([[1, 1], [0.1, 0.01], [0.1, 'no float']])

        test_sol_mat(sol_mat_1)
        test_sol_mat(sol_mat_2)
        test_sol_mat(sol_mat_3)

        # Check if different sol matrix sizes makes a problem
        sol_mat_4a = np.mat([[1, 1], [0.1, 0.01], [1, 1], [0.1, 0.01]])    # Wrong column size
        sol_mat_4b = np.mat([[1, 1], [0.1, 0.01]])                         # Wrong column size
        sol_mat_5a = np.mat([[1, 1], [0.1, 0.01], [0.1, 0.001, 0.002]])    # Wrong row size
        sol_mat_5b = np.mat([[1, 1], [0.1, 0.01], [0.1]])                  # Wrong row size

        test_sol_mat(sol_mat_4a)
        test_sol_mat(sol_mat_4b)
        test_sol_mat(sol_mat_5a)
        test_sol_mat(sol_mat_5b)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """
        self.failUnlessEqual(self.test_model.get_type(), 'PLSR')

    def test_get_regress_coeff(self):
        """ Tests whether the function PLSRMetaModel.get_regress_coeff returns the right values

        :return: The meta-model regression coefficient tested
        """
        self.failUnlessEqual(self.test_model.get_regress_coeff().all(), self.__sol_mat[1:].all())

    def test_get_output_const(self):
        """ Tests whether the function PLSRMetaModel.get_output_const returns the right values

        :return: The meta-model output constants tested
        """
        self.failUnlessEqual(self.test_model.get_output_const().all(), self.__sol_mat[0].all())

    def test_modify_input(self):
        """ Tests if the input is correctly modified

        :return: The modified input tested
        """

        # Since the output is the same for the first and the second, this is just a check if input is the same as output
        raw_input = np.mat([1, 0.4])
        self.failUnlessEqual(self.test_model.modify_input(raw_input).all(), raw_input.all())

    def test_calculate_output(self):
        """ Tests if the output is calculated correctly

        :return: The calculated output tested
        """

        # Calculate the right output first and then compare it with the output from the function
        input_par = np.mat([[1.1], [0.4]])
        output_par = np.zeros((1, 2))

        # First output value
        output_par[0, 0] = self.__sol_mat[0, 0]
        output_par[0, 0] += self.__sol_mat[1, 0] * input_par[0] + self.__sol_mat[2, 0] * input_par[1]

        # Second output value
        output_par[0, 1] = self.__sol_mat[0, 1]
        output_par[0, 1] += self.__sol_mat[1, 1] * input_par[0] + self.__sol_mat[2, 1] * input_par[1]

        self.failUnlessEqual(self.test_model.calculate_output(input_par).all(), output_par.all())


class TestDLUMetaModel(TestAbstractModel):
    """ A class to test the DLU meta-model structure

    """

    # Construct an example DLU meta-model
    __input_data = np.mat([[0.75, 0.25], [0.75, 0.35], [1.25, 0.25], [1.25, 0.35]])
    __output_data = np.mat([[1.75, 1.25], [1.75, 1.35], [2.25, 1.25], [2.25, 1.35]])
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    test_model = Models.DLUMetaModel(__input_data, __output_data, __in_par_intervals, __in_par_means,
                                       __in_par_variances, __out_par_intervals, __out_par_means, __out_par_variances)

    def test_initialization(self):
        """ A test for the initialization of the DLU MetaModel

        :return: A DLU meta-model tested
        """

        def test_databases(input_data, output_data):
            self.assertRaises(TypeError, Models.AbstractModel, input_data, output_data, self.__in_par_means, self.__in_par_means,
                              self.__in_par_variances, self.__out_par_intervals, self.__out_par_means,
                              self.__out_par_variances)

        # Check if not the proper input for the input database is used
        input_data_1 = 'not a list'
        input_data_2 = np.mat([[1, 1], [0.1, 0.01], 'not a list'])
        input_data_3 = np.mat([[1, 1], [0.1, 0.01], [0.1, 'no float']])

        test_databases(input_data_1, self.__output_data)
        test_databases(input_data_2, self.__output_data)
        test_databases(input_data_3, self.__output_data)

        # Check if different input database sizes makes a problem
        input_data_4a = np.mat([[1, 1], [0.1, 0.01], [1, 1], [0.1, 0.01]])          # Wrong column size
        input_data_4b = np.mat([[1, 1], [0.1, 0.01]])                               # Wrong column size
        input_data_5a = np.mat([[1, 1, 1], [0.1, 0.01, 1], [0.1, 0.001, 0.002]])    # Wrong row size
        input_data_5b = np.mat([[1], [0.1], [0.1]])                                 # Wrong row size

        test_databases(input_data_4a, self.__output_data)
        test_databases(input_data_4b, self.__output_data)
        test_databases(input_data_5a, self.__output_data)
        test_databases(input_data_5b, self.__output_data)

        # Check if not the proper output for the output database is used
        output_data_1 = 'not a list'
        output_data_2 = np.mat([[1, 1], [0.1, 0.01], 'not a list'])
        output_data_3 = np.mat([[1, 1], [0.1, 0.01], [0.1, 'no float']])

        test_databases(self.__input_data, output_data_1)
        test_databases(self.__input_data, output_data_2)
        test_databases(self.__input_data, output_data_3)

        # Check if different output database sizes makes a problem
        output_data_4a = np.mat([[1, 1], [0.1, 0.01], [1, 1], [0.1, 0.01]])         # Wrong column size
        output_data_4b = np.mat([[1, 1], [0.1, 0.01]])                              # Wrong column size
        output_data_5a = np.mat([[1, 1, 1], [0.1, 0.01, 1], [0.1, 0.001, 0.002]])   # Wrong row size
        output_data_5b = np.mat([[1], [0.1], [0.1]])                                # Wrong row size

        test_databases(self.__input_data, output_data_4a)
        test_databases(self.__input_data, output_data_4b)
        test_databases(self.__input_data, output_data_5a)
        test_databases(self.__input_data, output_data_5b)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """
        self.failUnlessEqual(self.test_model.get_type(), 'DLU')

    def test_modify_input(self):
        """ Tests if the input is correctly modified

        :return: The modified input tested
        """

        # Since the output is the same for the first and the second, this is just a check if input is the same as output
        raw_input = np.mat([1, 0.4])
        self.failUnlessEqual(self.test_model.modify_input(raw_input).all(), raw_input.all())

    def test_calculate_output(self):
        """ Tests if the output is calculated correctly

        :return: The calculated output tested
        """

        # Calculate the right output first and then compare it with the output from the function
        input_par_1 = np.mat([[1.1], [0.4]])
        output_par_1 = np.mat([[2.25], [1.35]])
        input_par_2 = np.mat([[0.6], [0.2]])
        output_par_2 = np.mat([[1.75], [1.25]])

        self.failUnlessEqual(self.test_model.calculate_output(input_par_1).all(), output_par_1.all())
        self.failUnlessEqual(self.test_model.calculate_output(input_par_2).all(), output_par_2.all())


class TestModelDecorator(TestAbstractModel):
    """ All test for the modelDecorator to see if this decorator works as it should

    """

    # Construct an example Abstract meta-model
    __in_par_intervals = np.mat([[0.5, 1.5], [0.2, 0.4]])
    __in_par_means = np.mat([1.2, 0.3])
    __in_par_variances = np.mat([0.1, 0.001])
    __out_par_intervals = np.mat([[1.5, 2.5], [1.2, 1.4]])
    __out_par_means = np.mat([2.2, 1.3])
    __out_par_variances = np.mat([0.1, 0.001])

    test_model_basic = Models.AbstractModel(__in_par_intervals, __in_par_means, __in_par_variances, __out_par_intervals,
                                        __out_par_means, __out_par_variances)

    test_model = Models.ModelDecorator(test_model_basic)

    def test_get_type(self):
        """ Tests whether the function AbstractModel.get_type returns the right type

        :return: The meta-model type tested
        """
        self.failUnlessEqual(self.test_model.get_type(), 'Decorated Abstract')


class TestInputDecorator(TestModelDecorator):
    """

    """