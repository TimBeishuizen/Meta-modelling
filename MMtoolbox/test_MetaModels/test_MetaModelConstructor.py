from unittest import TestCase
from MetaModels import MetaModelConstructor as MMC
import numpy as np


class test_constructors(TestCase):
    """ A test case for all constructors of the MetaModelConstructor file

        """

    __database_1 = np.mat([[1.3, 4.2, 5.6, 3.5], [2.5, 4.1, 6.4, 0.3], [3.5, 2.3, 1.4, 3.9]])
    __database_2 = np.mat([[0.5, 0.3, 1.4], [0.7, 0.8, 1.1], [0.8, 0.2, 1.3]])

    def test_DLU_constructor(self):
        """ Tests if the DLU constructor works
        
        :return: Tests if the DLU constructor works
        """

        DLU ={'type': "DLU"}

        DLU_meta_model = MMC.construct_meta_model(self.__database_1, self.__database_2, DLU)

        np.testing.assert_equal(DLU_meta_model.mm_type, "DLU")

    def test_PLSR_constructor(self):
        """ Tests if the PLSR constructor works

        :return: Tests if the PLSR constructor works
        """

        PLSR = {'type': "PLSR", 'n_comp': 4}

        PLSR_meta_model = MMC.construct_meta_model(self.__database_1, self.__database_2, PLSR)

        np.testing.assert_equal(PLSR_meta_model.mm_type, "PLSR")

        print(PLSR_meta_model.calculate_output(np.mat([[3,2,1,3]])))

    def test_PLSR_constructor2(self):
        """ Tests if the PLSR constructor works2

        :return: Tests if the PLSR constructor works
        """

        in_dat = np.mat([[3.4, 2.1, 6.7, 5.6],[ 5.8, 9.1, 3.7, 8.2],[2.8, 3.1, 7.7, 1.2]])
        out_dat = np.mat([[2.0, 1.7, 3.1, 4.2],[ 6.8, 4.1, 5.4, 7.8], [ 2.8, 1.1, 4.4, 5.1]])

        PLSR = {'type': "PLSR"}

        PLSR_meta_model = MMC.construct_meta_model(in_dat, out_dat, PLSR)

        np.testing.assert_equal(PLSR_meta_model.mm_type, "PLSR")

        print(PLSR_meta_model.calculate_output(np.mat([[3,2,1,3]])))

    def test_DLU_Poly_constructor(self):
        """ Tests if the DLU constructor works in combination with poly

        :return: Tests if the DLU constructor works in combination with poly
        """

        DLU = {'type': "DLU", "input": "Polynomial"}

        DLU_meta_model = MMC.construct_meta_model(self.__database_1, self.__database_2, DLU)

        np.testing.assert_equal(DLU_meta_model.get_type(), "Polynomial Input DLU")

        print(DLU_meta_model.simulate(np.mat([[1, 4, 5, 3]])))

    def test_DLU_Mod_constructor(self):
        """ Tests if the DLU constructor works in combination with mod

                :return: Tests if the DLU constructor works in combination with mod
                """

        DLU = {'type': "DLU", "input": "Modified", 'specs': ['log', 'sqr', 'root']}

        DLU_meta_model = MMC.construct_meta_model(self.__database_1, self.__database_2, DLU)

        np.testing.assert_equal(DLU_meta_model.get_type(), "Modified Input DLU")

        print(DLU_meta_model.simulate(np.mat([[3, 2, 1, 3]])))

    def test_PLSR_Poly_constructor(self):
        """ Tests if the PLSR constructor works in combination with poly

                :return: Tests if the PLSR constructor works in combination with poly
                """

        PLSR = {'type': "PLSR", "input": "Polynomial"}

        PLSR_meta_model = MMC.construct_meta_model(self.__database_1, self.__database_2, PLSR)

        np.testing.assert_equal(PLSR_meta_model.get_type(), "Polynomial Input PLSR")

        print(PLSR_meta_model.simulate(np.mat([[1, 4, 5, 3]])))

    def test_PLSR_Mod_constructor(self):
        """ Tests if the DLU constructor works in combination with mod

                :return: Tests if the DLU constructor works in combination with mod
                """

        PLSR = {'type': "PLSR", "input": "Modified", 'specs': ['sqr']}

        PLSR_meta_model = MMC.construct_meta_model(self.__database_1, self.__database_2, PLSR)

        np.testing.assert_equal(PLSR_meta_model.get_type(), "Modified Input PLSR")

        print(PLSR_meta_model.simulate(np.mat([[3, 2, 1, 3]])))

class test_save_load_files(TestCase):
    """ A class made to test the save and load files in the constructor
    
    """

    __database_1 = np.mat([[1.3, 4.2, 5.6, 3.5], [2.5, 4.1, 6.4, 0.3], [3.5, 2.3, 1.4, 3.9]])
    __database_2 = np.mat([[0.5, 0.3, 1.4], [0.7, 0.8, 1.1], [0.8, 0.2, 1.3]])

    def create_meta_model(self):
        """ Constructs a meta-model for saving and loading
        
        :return: A made meta-model
        """

        return MMC.construct_meta_model(self.__database_1, self.__database_2, {"type": "DLU", "input": "Basic"})

    def test_save_meta_model(self):
        """ A test to save a meta-model - visual test
        
        :return: A result to the test
        """

        meta_model = self.create_meta_model()

        MMC.save_meta_model(meta_model, "save_test", False)

    def test_load_meta_model(self):
        """ A test to load a meta-model -visual test
        
        :return: A result to the test
        """

        meta_model = self.create_meta_model()

        new_meta_model = MMC.load_meta_model("save_test")

