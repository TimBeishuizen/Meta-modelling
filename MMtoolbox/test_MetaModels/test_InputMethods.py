from unittest import TestCase
from MetaModels import InputMethods as IM
from MetaModels import ConstructionMethods as CM
import numpy as np

class test_InputMethods(TestCase):
    """ Tests the input methods of Input methods
    
    """

    def test_polynomialize_database(self):
        """ Tests if polynomialize database is working fine
        
        :return: Test result
        """

        input_database = np.mat([[100.1, 0.4, 0.6], [3.4, 7.1, 0.9], [5.3, 1.8, 7.3]])
        input_line_1 = input_database[0]
        input_line_2 = input_database[1]
        input_line_3 = input_database[2]

        input_means = CM.compute_means(input_database)
        input_variances = CM.compute_variances(input_database)

        mod_line_1 = IM.polynomialize_input(input_line_1, input_means, input_variances)
        mod_line_2 = IM.polynomialize_input(input_line_2, input_means, input_variances)
        mod_line_3 = IM.polynomialize_input(input_line_3, input_means, input_variances)

        mod_database = (np.zeros([input_database.shape[0], mod_line_1.shape[1]]))
        mod_database[0] = mod_line_1
        mod_database[1] = mod_line_2
        mod_database[2] = mod_line_3
        print(mod_database)

        np.testing.assert_equal(IM.polynomialize_database(input_database), mod_database)

    def test_polynomialize_input(self):
        """ Tests if the input is correctly modified

                :return: A result of the test if it is correctly modified
                """

        # The raw and modified input parameters
        input_database = np.mat([[1.1, 0.4, 0.6], [5.6, 4.3, 3.2], [2.6, 0.6, 1.8]])
        input_par_1 = np.mat([[1.1, 0.4, 0.6]])
        in_par_means = CM.compute_means(input_database)
        in_par_variances = CM.compute_variances(input_database)

        nr_par = int(input_par_1.shape[1])
        nr_mod_par = int(2 * nr_par + (nr_par * nr_par - 1) / 2) - 1
        mod_input_par_1 = np.mat(np.zeros(nr_mod_par))
        stand_par = IM.standardize_input(input_par_1, in_par_means, in_par_variances)
        mod_input_par_1[0, range(nr_par)] = stand_par[0]

        print(stand_par)
        # Add all terms of the polynomial input parameters to the modified input parameters
        next_par = nr_par
        for i in range(nr_par):
            for j in range(i, nr_par):
                mod_input_par_1[0, next_par] = mod_input_par_1[0, i] * mod_input_par_1[0, j]
                next_par += 1

        np.testing.assert_array_equal(IM.polynomialize_input(input_par_1, in_par_means, in_par_variances), mod_input_par_1)

    def test_standardize_input(self):
        """ Tests if the input is correctly standardized

        :return: A result of the test if it is correctly standardized
        """


        # The raw and modified input parameters
        input_database = np.mat([[1.1, 0.4], [5.6, 4.3], [2.6, 0.6]])
        input_par_1 = np.mat([[1.1, 0.4]])
        in_par_means = CM.compute_means(input_database)
        in_par_variances = CM.compute_variances(input_database)

        mean_par_1 = np.subtract(input_par_1, in_par_means)
        mod_par_1 = np.divide(mean_par_1, np.sqrt(in_par_variances))

        np.testing.assert_array_equal(IM.standardize_input(input_par_1, in_par_means, in_par_variances), mod_par_1)

    def test_modify_database(self):
        """ test is the database is correctly modified
        
        :return: Answer to the test
        """

        input_database = np.mat([[1.1, 0.4], [5.6, 4.3], [2.6, 0.6]])
        modifiers = ["log", 'sqr', 'root', 'inv', 'exp', 'sin', 'cos', 'tan']

        mod_line_1 = IM.modify_input(input_database[0], modifiers)
        mod_line_2 = IM.modify_input(input_database[1], modifiers)
        mod_line_3 = IM.modify_input(input_database[2], modifiers)

        mod_database = np.zeros([input_database.shape[0],mod_line_1.shape[1]])

        mod_database[0] = mod_line_1
        mod_database[1] = mod_line_2
        mod_database[2] = mod_line_3

        np.testing.assert_array_equal(IM.modify_database(input_database, modifiers), mod_database)

    def test_modify_input(self):
        """ tests if the input is correctly modified
        
        :return: Answer to the test
        """

        # The raw and modified input parameters
        input_par_1 = np.mat([[1.1, 0.4]])
        modifiers = ["log",'sqr', 'root', 'inv', 'exp', 'sin', 'cos', 'tan']

        nr_par = int(input_par_1.shape[1])
        mod_input_par_1 = np.mat(np.zeros((len(modifiers) + 1) * nr_par))
        mod_input_par_1[0, range(nr_par)] = input_par_1[0]

        for i in range(len(modifiers)):
            if modifiers[i] == 'log':
                temp_par = np.log(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif modifiers[i] == 'sqr':
                temp_par = np.square(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par
            elif modifiers[i] == 'root':
                temp_par = np.sqrt(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par
            elif modifiers[i] == 'inv':
                temp_par = np.power(input_par_1, -1.)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif modifiers[i] == 'exp':
                temp_par = np.exp(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif modifiers[i] == 'sin':
                temp_par = np.sin(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif modifiers[i] == 'cos':
                temp_par = np.cos(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]
            elif modifiers[i] == 'tan':
                temp_par = np.tan(input_par_1)
                mod_input_par_1[0, 2 * (i + 1):2 * (i + 1) + 2] = temp_par[0]

        np.testing.assert_array_equal(IM.modify_input(input_par_1, modifiers), mod_input_par_1)
