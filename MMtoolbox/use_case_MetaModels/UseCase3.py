from MetaModels import MetaModelConstructor as MMC

""" Use Case 3 - Retrieve values from a meta-model

"""

# Load the earlier made meta-model
MM = MMC.load_meta_model('UseCaseMetaModel')

# Then prints all values of the meta-models to get more insight in the data
print(MM.get_type())
print(MM.get_in_par_intervals())
print(MM.get_in_par_means())
print(MM.get_in_par_variances())
print(MM.get_outpar_intervals())
print(MM.get_out_par_means())
print(MM.get_out_par_variances())
print(MM.get_output_const())
print(MM.get_regress_coeff())

