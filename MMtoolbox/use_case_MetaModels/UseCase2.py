import numpy as np
from MetaModels import MetaModelConstructor as MMC

""" Use Case 2 - Simulate with a meta-model

"""

# Load the earlier made meta-model
PLSR_MM = MMC.load_meta_model('UseCaseMetaModel')

# Find test values with known outcome to test it with
test_values = np.mat([0.00600387, 0.01150516, 0.00016985, 0.00192667])
known_outcome = np.mat([[ 0.7854397,   1.0055758,  1.0368996,  0.77191588, 0.60249322,  0.52795852,
   0.49579548,  0.47082445,  0.45773433,  0.44649127,  0.43574831,  0.42527118,
   0.40506458,  0.39054897,  0.376558,    0.36306583,  0.34582864,  0.32940859,
   0.31376478,  0.29525228,  0.27784048,  0.2614507,   0.24602328,  0.76306239,
   0.59543806,  0.41933948,  0.30676955,  0.27745082,  0.26640204,  0.260563,
   0.2530352,  0.24678511,  0.24083199,  0.23504354,  0.22939523,  0.21850215,
   0.21067376,  0.20312629,  0.19584895,  0.18655014,  0.17769265,  0.16925494,
   0.15926983,  0.1498751,   0.14103327,  0.1327128 ]])

# Simulate with the meta-model
print(PLSR_MM.simulate(test_values))

# Retrieve answer. Answer is known to be bad on 23 first outcomes and good on next 23 outcomes
print(PLSR_MM.simulate(test_values) - known_outcome)