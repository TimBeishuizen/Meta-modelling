from unittest import TestCase
from MetaModels import MetaModelConstructor as MMC
import numpy as np


class test_side_functions(TestCase):
    """ A test case for all side functions of the MetaModelConstructor file

    """

    __database_1 = np.mat([[1., 4., 5., 3.], [2., 4., 6., 0.], [3., 2., 1., 3.], [5., 3., 4., 2.]])
    __database_2 = np.mat([[0.5, 0.3, 1.4], [0.7, 0.8, 1.1], [0.8, 0.2, 1.3]])

    __output_database_1 = np.mat([[0.5, 1.4], [0.8, 1.1], [0.8, 1.3], [1.3, 1.1]])

    __matrix_1 = np.mat([[1, 0.5], [2, 6]])

    def test_compute_means(self):
        """ Tests if the function compute_means works correctly

        :return: Tests if the function compute_means works correctly
        """

        # Database 1
        means_1 = np.mat([[2.75, 3.25, 4., 2.]])
        np.testing.assert_array_equal(MMC.compute_means(self.__database_1), means_1)

        # Database 2
        means_2 = np.mat([2.0/3, 1.3/3, 3.8/3])
        np.testing.assert_array_equal(MMC.compute_means(self.__database_2), means_2)

    def test_compute_variances(self):
        """ Tests if the function compute_variances works correctly

        :return: Tests if the function compute_variances works correctly
        """

        # Database 1
        means_1 = MMC.compute_means(self.__database_1)
        temp_var = np.zeros(self.__database_1.shape)
        for i in range(self.__database_1.shape[0]):
            temp_var[i] = np.square(np.subtract(self.__database_1[i], means_1))
        variances_1 = np.mat(np.mean(temp_var, 0))

        np.testing.assert_array_equal(MMC.compute_variances(self.__database_1), variances_1)

        # Database 2
        means_2 = MMC.compute_means(self.__database_2)
        temp_var = np.zeros(self.__database_2.shape)
        for i in range(self.__database_2.shape[0]):
            temp_var[i] = np.square(np.subtract(self.__database_2[i], means_2))
        variances_2 = np.mat(np.mean(temp_var, 0))

        np.testing.assert_array_equal(MMC.compute_variances(self.__database_2), variances_2)

    def test_compute_intervals(self):
        """ Tests if the function test_compute_intervals works correctly

        :return: Tests if the function test_compute_intervals works correctly
        """

        # Database 1
        intervals_1 = np.mat(np.zeros((self.__database_1.shape[1], 2)))
        intervals_1[:, 0] = np.inf

        for i in range(self.__database_1.shape[0]):
            for j in range(self.__database_1.shape[1]):
                if intervals_1[j, 0] > self.__database_1[i, j]:
                    intervals_1[j, 0] = self.__database_1[i, j]
                if intervals_1[j, 1] < self.__database_1[i, j]:
                    intervals_1[j, 1] = self.__database_1[i, j]

        np.testing.assert_array_equal(MMC.compute_intervals(self.__database_1), intervals_1)

        # Database 2
        intervals_2 = np.mat(np.zeros((self.__database_2.shape[1], 2)))
        intervals_2[:, 0] = np.inf

        for i in range(self.__database_2.shape[0]):
            for j in range(self.__database_2.shape[1]):
                if intervals_2[j, 0] > self.__database_2[i, j]:
                    intervals_2[j, 0] = self.__database_2[i, j]
                if intervals_2[j, 1] < self.__database_2[i, j]:
                    intervals_2[j, 1] = self.__database_2[i, j]

        print(intervals_2)
        print(MMC.compute_intervals(self.__database_2))

        np.testing.assert_array_equal(MMC.compute_intervals(self.__database_2), intervals_2)

    def test_compute_norm(self):
        """ Tests if the method compute_norm does what it should do

        :return: Tests if the method compute_norm does what it should do
        """

        # Norm 1
        norm_1 = 6.3745

        self.assertAlmostEqual(MMC.compute_norm(self.__matrix_1), norm_1, delta=0.0001)

        # Norm 2
        norm_2 = 12.6164

        self.assertAlmostEqual(MMC.compute_norm(self.__database_1), norm_2, delta=0.0001)

    def test_sing_val_decomp(self):
        """ Tests if the function sing_val_decomps returns the proper output

        :return: Tests if the function sing_val_decomps returns the proper output
        """

        # SVD 1
        u_1 = np.mat([[-0.5380], [-0.5566], [-0.3066], [-0.5538]])
        s_1 = 12.6164
        v_1 = np.mat([-0.4233, -0.5273, -0.6778, -0.2886])

        dic_1 = MMC.compute_sing_val_decomp(self.__database_1)
        u_r_1 = dic_1.get('U')
        s_r_1 = dic_1.get('s')
        v_r_1 = dic_1.get('V')

        np.testing.assert_array_almost_equal(u_r_1, u_1, decimal=4)
        self.assertAlmostEqual(s_r_1, s_1, delta=0.0001)
        np.testing.assert_array_almost_equal(v_r_1, v_1, decimal=4)

    def test_update_orth_basis(self):
        """ Test if the method update_orth_basis works properly

        :return: Test if the method update_orth_basis works properly
        """

        # updated basis 1
        orth_mat_1 = np.mat([[0.5, 0.3, 0, 0], [0.7, 0.8, 0, 0], [0.8, 0.2, 0, 0], [0.5, 0.5, 0, 0]])
        updated_orth_mat_1 = np.mat([[0.5, 0.3, 0.3350, 0], [0.7, 0.8, 0.0741, 0],
                                     [0.8, 0.2, -0.9391, 0], [0.5, 0.5, 0.0190, 0]])

        np.testing.assert_array_almost_equal(MMC.update_orth_basis(orth_mat_1, self.__database_1[:, 2], 3),
                                             updated_orth_mat_1, decimal=4)

        # updated basis 2
        orth_mat_2 = np.mat([[0.5, 0.3, 0], [0.7, 0.8, 0], [0.8, 0.2, 0]])
        updated_orth_mat_2 = np.mat([[0.5, 0.3, 0.8446], [0.7, 0.8, -0.0743], [0.8, 0.2, -0.5302]])

        print(MMC.update_orth_basis(orth_mat_2, self.__database_2[:, 2], 3))

        np.testing.assert_array_almost_equal(MMC.update_orth_basis(orth_mat_2, self.__database_2[:, 2], 3),
                                             updated_orth_mat_2, decimal=4)

    def test_update_cov(self):
        """ Tests if the method update_cov works correctly

        :return: Tests if the method update_cov works correctly
        """

        # updated cov 1
        cov_1 = np.mat([[4, 4, 6, 0, 4],  [4, 3, 4, 2, 6], [2, 4, 5, 3, 9], [5, 2, 1, 3, 8]])
        updated_orth_mat_1 = np.mat([[0.5, 0.3, 0.3350, 0], [0.7, 0.8, 0.0741, 0],
                                     [0.8, 0.2, -0.9391, 0], [0.5, 0.5, 0.0190, 0]])
        upd_cov_1 = np.mat([[-2.5517, -0.4725, 0.3263, -2.1880, -4.3192],
                            [-8.0033, -6.1499, -7.3326, -3.4826, -12.6583],
                            [-6.6559, -4.8357, -5.7664, -3.3317, -10.6514],
                            [-3.0575, -4.2259, -6.7139, -0.7438, -4.7139]])

        np.testing.assert_array_almost_equal(MMC.update_cov(cov_1, updated_orth_mat_1, 3), upd_cov_1, decimal=3)

        # updated cov 2
        cov_2 = np.mat([[3, 2, 4, 4, 0],  [ 1, 8, 6, 4, 5], [9, 2, 3, 2, 3]])
        updated_orth_mat_2 = np.mat([[0.5, 0.3, 0.8446], [0.7, 0.8, -0.0743], [0.8, 0.2, -0.5302]])
        upd_cov_2 = np.mat([[-0.7960, -4.3493, -3.4144, -2.3481, -2.6718],
                            [-8.6273, -3.6563, -5.1565, -4.1039, -3.0198],
                            [-0.3869, -6.0225, -4.5226, -3.0600, -3.6302]])

        np.testing.assert_array_almost_equal(MMC.update_cov(cov_2, updated_orth_mat_2, 3), upd_cov_2, decimal=3)

    def test_SIMPLS(self):
        """ This tests the method SIMPLS if it works correctly

        :return: This tests the method SIMPLS if it works correctly
        """

        # database 1
        regress_coeff = np.mat([[0.4751, 0.0695], [-0.0081, 0.0113], [-0.2203, -0.0122], [0.2846, 0.0429]])

        np.testing.assert_array_almost_equal(MMC.SIMPLS(self.__database_1, self.__output_database_1, 2),
                                             regress_coeff, decimal=4)

    def test_center_database(self):
        """ Tests if method test_center_database works correctly

        :return: Tests if method test_center_database works correctly
        """

        # database 1
        cent_data_1 = np.mat(np.zeros(self.__database_1.shape))
        means_1 = MMC.compute_means(self.__database_1)
        for i in range(self.__database_1.shape[0]):
            cent_data_1[i, :] = np.divide(self.__database_1[i, :], means_1)

        np.testing.assert_array_equal(MMC.center_database(self.__database_1), cent_data_1)

    def test_PLSR(self):
        """ Tests if the function PLSR works porperly

        :return: Tests if the function PLSR works porperly
        """

        # database 1
        print(MMC.PLSR(self.__database_1, self.__output_database_1, 2))