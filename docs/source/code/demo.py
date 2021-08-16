import pandas as pd

from lassobbn.learn import do_learn, to_bbn
from pybbn.pptc.inferencecontroller import InferenceController

# Step 1. read the data
df = pd.read_csv('./data/data-binary.csv')

# Step 2. learn the structure and parameters
json_data = do_learn(df)
import json
print(json.dumps(json_data, indent=1))

# Step 3. convert to a Py-BBN instance
bbn = to_bbn(json_data)

# Step 4. convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# print the posterior probabilities
for node, posteriors in join_tree.get_posteriors().items():
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p}')

# inference should print the following
# a : 0=0.18930, 1=0.81070
# b : 0=0.80290, 1=0.19710
# c : 0=0.68690, 1=0.31310
# d : 0=0.79520, 1=0.20480
# e : 0=0.74160, 1=0.25840
