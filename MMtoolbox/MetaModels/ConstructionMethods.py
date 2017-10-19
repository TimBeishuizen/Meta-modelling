import numpy as np
from MetaModels import RobustnessMethods as RM


def PLSR(input_data, output_data, n_comp):
    """ Performs partial least squares regression on the input and output data to create a meta-model

    :param input_data: The input database used for PLSR
    :param output_data: The output database used for PLSR
    :param n_comp: The number of principal components used in PLSR
    :return: The solution matrix by PLSR
    """

    cent_input_data = center_database(input_data)
    cent_output_data = center_database(output_data)

    regress_coeff = SIMPLS(cent_input_data, cent_output_data, n_comp)

    input_mean = compute_means(input_data)
    output_mean = compute_means(output_data)

    sol_mat = np.mat(np.zeros((regress_coeff.shape[0] + 1, regress_coeff.shape[1])))
    sol_mat[1:, :] = regress_coeff[:, :]
    sol_mat[0, :] = output_mean - input_mean * regress_coeff

    return sol_mat


def center_database(database):
    """ Centers the data around their mean

    :param database: The database that needs to be centered
    :return: A centered database
    """

    means = compute_means(database)
    return np.divide(database, means)


def compute_sing_val_decomp(matrix):
    """ Computes the singular value decomposition of a matrix

    :param matrix: The matrix that needs singular value decomposition
    :return: The eigenvectors U and V and square root of the eigenvalues s
    """

    u, s, v = np.linalg.svd(matrix)
    first_eig_vec_u = np.mat(u[:, 0])
    first_eig_val_s = s[0]
    first_eig_vec_v = np.mat(v[0, :])

    return {'U': first_eig_vec_u, 's': first_eig_val_s, 'V': first_eig_vec_v}


def compute_norm(matrix):
    """ Computes the norm of a matrix

    :param matrix: The matrix the norm is needed from
    :return: The norm of the matrix
    """

    return np.linalg.norm(matrix, 2)


def update_orth_basis(orth_basis, curr_loadings, index):
    """ Updates the orthonormal basis with the current loadings. This is done with modified Gram Schmidt

    :param orth_basis: The orthonormal basis not updated yet
    :param curr_loadings: The loadings the orthonormal basis
    :param index: The to be updated row of the orthonormal basis
    :return:  The updated orthonormal basis
    """

    def update_loadings(orth_basis, curr_loadings, index):
        """ Calculate the updated loadings for the new updated orthonormal basis

        :param orth_basis: The orthonormal basis
        :param curr_loadings: The current loadings not updated yet
        :param index: The row of the current loadings
        :return: The updated loadings
        """

        upd_loadings = curr_loadings

        for i in range(index - 1):
            old_loadings = orth_basis[:, i]
            upd_loadings = np.subtract(upd_loadings, old_loadings * (np.transpose(old_loadings) * upd_loadings))

        return upd_loadings

    # Update twice for Gram Schmidt
    updated_loadings = update_loadings(orth_basis, curr_loadings, index)
    updated_loadings = update_loadings(orth_basis, updated_loadings, index)


    # Divide by norm
    updated_loadings /= compute_norm(updated_loadings)
    orth_basis[:, index - 1] = updated_loadings

    return orth_basis


def update_cov(cov_matrix, orth_matrix, index):
    """ Updates the covariance matrix to the new orthonormal basis

    :param cov_matrix: The ocvariance matrix to be updated
    :param orth_matrix: The orthonormal matrix used for the update
    :param index: The index of the latest additions on the cov matrix
    :return: The update of the covariance matrix
    """

    updated_cov = cov_matrix - orth_matrix[:, index - 1] * (np.transpose(orth_matrix[:, index - 1]) * cov_matrix)
    updated_cov -= orth_matrix[:, range(index - 1)] * (np.transpose(orth_matrix[:, range(index - 1)]) * updated_cov)

    return updated_cov


def SIMPLS(input_data, output_data, n_comp):
    """ A method called SIMPLS to calculate the weights for a solution matrix

    :param input_data: the input database for the meta-model
    :param output_data: the output database for the meta-model
    :param n_comp: the number of components for PCA used
    :return: The regression coefficients of the matrix
    """

    # First initialize all made values
    samp = input_data.shape[0]
    input_size = input_data.shape[1]
    output_size = output_data.shape[1]

    input_loadings = np.mat(np.zeros((input_size, n_comp)))
    output_loadings = np.mat(np.zeros((n_comp, output_size)))
    weights = np.mat(np.zeros((input_size, n_comp)))
    input_scores = np.mat(np.zeros((samp, n_comp)))

    orth_basis = np.mat(np.zeros((input_size, n_comp)))
    cov_mat = np.transpose(input_data) * output_data

    # The weights are made for every separate component
    for i in range(n_comp):
        # Perform singular value decomposition
        svd_dict = compute_sing_val_decomp(cov_mat)
        eig_vec_u = svd_dict.get('U')
        eig_val_s = svd_dict.get('s')
        eig_vec_v = svd_dict.get('V')

        # Calculate the scores and loadings
        curr_scores = input_data * eig_vec_u
        norm_scores = compute_norm(curr_scores)
        curr_scores /= norm_scores
        input_scores[:, i] = curr_scores

        curr_loadings = np.transpose(input_data) * curr_scores
        input_loadings[:, i] = curr_loadings

        output_loadings[i, :] = np.transpose(eig_val_s * np.transpose(eig_vec_v) / norm_scores)

        weights[:, i] = eig_vec_u / norm_scores

        # Update the orthonormal basis and change the covariance matrix according to this new orthonormal basis
        orth_basis = update_orth_basis(orth_basis, curr_loadings, i)
        cov_mat = update_cov(cov_mat, orth_basis, i)

    print(weights.shape)
    print(output_loadings.shape)

    regress_coeff = weights * output_loadings



    return regress_coeff


def compute_means(database):
    """ Computes the means of the database

    :param database: The database the means are needed of
    :return: The means of the database
    """

    RM.check_if_matrix(database, 'Currently using database using for the means')

    return np.mean(database, 0)


def compute_variances(database):
    """ Computes the variances of the database

    :param database: The database the variances are needed of
    :return: The variances of the database
    """

    RM.check_if_matrix(database, 'Currently using database using for the variances')

    return np.var(database, 0)


def compute_intervals(database):
    """ Computes the intervals of the database

    :param database: The database the intervals are needed of
    :return: The intervals of the database
    """

    RM.check_if_matrix(database, 'Currently using database using for the intervals')

    intervals = np.mat(np.zeros((database.shape[1], 2)))
    intervals[:, 0] = np.transpose(np.min(database, 0))
    intervals[:, 1] = np.transpose(np.max(database, 0))

    return intervals