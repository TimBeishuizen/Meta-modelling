import pickle as pk
from MetaModels import MetaModel as MM
from MetaModels import MetaModelDecorator as MMD
from MetaModels import ConstructionMethods as CM
from MetaModels import RobustnessMethods as RM
from MetaModels import InputMethods as IM


def construct_meta_model(input_data, output_data, meta_model_specifications):
    """ The construction of the meta-models. This uses the input and output database to create a way to use the
    meta-models.

    :param input_data: A matrix with the input database. The first dimension being all the samples and the second
    dimension all input parameters
    :param output_data: A matrix with the output database. The first dimension consists of all the samples and the
    second dimensions consists of all output parameters
    :param meta_model_specifications: All specifications for the meta-models, being a dictionary with the following possible
     specifications:
        - type: type of meta-model. Either 'DLU' or 'PLSR'. In case of 'PLSR' the next entry can be the number of
                components if another number of components than all is desired
        if not the regular amount
        - input: type of input modifier. Either 'Basic', 'Polynomial' or 'Modified', by default 'basic'. If modified,
                the next entry should give the type of modifications: 'log', 'sqr', 'root', 'inv', 'exp', 'sin', 'cos',
                'tan'.
        - cluster: Whether clusters are included or not. Either 'cluster' or nothing. If 'cluster', the next entry
                should give the number of clusters
    :return: The correctly constructed meta-model
    """

    # First compute all extra input values
    input_intervals = CM.compute_intervals(input_data)
    input_means = CM.compute_means(input_data)
    input_variances = CM.compute_variances(input_data)

    # Then compute all extra output values
    output_intervals = CM.compute_intervals(output_data)
    output_means = CM.compute_means(output_data)
    output_variances = CM.compute_variances(output_data)

    # Check the specifications
    meta_model_type = meta_model_specifications['type']
    if "input" in meta_model_specifications:
        meta_model_input = meta_model_specifications["input"]
    else:
        meta_model_input = "Basic"

    # Check for changes in specifications
    if meta_model_input == "Polynomial":
        input_data = IM.polynomialize_database(input_data)
    elif meta_model_input == "Modified":
        input_data = IM.modify_database(input_data, meta_model_specifications["specs"])

    if meta_model_type == "DLU":
        meta_model = MM.DLUMetaModel(input_data, output_data, input_intervals, input_means, input_variances,
                                     output_intervals, output_means, output_variances)

    elif meta_model_type == "PLSR":
        if 'n_comp' in meta_model_specifications:
            n_comp = meta_model_specifications["n_comp"]
        else:
            n_comp = 3
        sol_mat = CM.PLSR(input_data, output_data, n_comp)
        meta_model = MM.PLSRMetaModel(sol_mat, input_intervals, input_means, input_variances,
                                      output_intervals, output_means, output_variances)
    else:
        raise ValueError("The name of the meta-model is not implemented")

    if meta_model_input == "Polynomial":
        meta_model = MMD.PolynomialInputDecorator(meta_model)

    elif meta_model_input == "Modified":
        meta_model = MMD.ModifiedInputDecorator(meta_model, meta_model_specifications["specs"])

    elif meta_model_input == "Basic":
        None
    else:
        raise ValueError("The name of the meta-model is not implemented")

    return meta_model


def save_meta_model(meta_model, file_name, rewrite):
    """ Saves the meta_model under the name file_name
    
    :param meta_model: The meta_model
    :param file_name: The file_name
    :param rewrite: Whether a file may be rewritten: "True" it may be, "False" or None,
    :return: File is saved
    """

    RM.check_if_meta_model(meta_model, "This meta_model")
    RM.check_if_string(file_name, "This file name")

    if not isinstance(rewrite, bool) or not rewrite:
        RM.check_if_file_exists(file_name, False)

    with open(file_name + ".pkl", 'wb') as output:
        pk.dump(meta_model, output)


def load_meta_model(file_name):
    """ Loads the meta_model under the name file_name

        :param file_name: The file_name
        :return: The loaded meta-model
        """

    RM.check_if_string(file_name, "This file name")
    #RM.check_if_file_exists(file_name, True)

    with open(file_name + ".pkl", 'rb') as input:
        filedict = pk.load(input)

    return filedict