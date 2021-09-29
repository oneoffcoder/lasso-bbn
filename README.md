![pybbn logo](https://lasso-bbn.readthedocs.io/en/latest/_images/logo-250x250.png)

# LASSO BBN

Learning Bayesian Belief Networks (BBNs) with LASSO. Example code is as below. 

```python
from lassobbn.learn import learn_parameters, learn_structure, to_bbn, to_join_tree, posteriors_to_df

# Step 1. Learn the structure
df_path = './data/data-binary.csv'
meta_path = './data/data-binary-complete.json'

parents = learn_structure(df_path, meta_path, n_way=2, ignore_neg_gt=-0.01, ignore_pos_lt=0.05)

# Step 2. Learn the parameters
d, g, p = learn_parameters(df_path, parents)

# Step 3. Get the BBN
bbn = to_bbn(d, g, p)

# Step 4. Get the Join Tree
jt = to_join_tree(bbn)

```

You can then use [Py-BBN](https://py-bbn.readthedocs.io/) to create a BBN and join tree (JT) instance and perform exact inference.

# Installation

```bash
pip install lassobbn
```

# Links

- [Code](https://github.com/oneoffcoder/lasso-bbn)
- [Documentation](https://lasso-bbn.readthedocs.io/en/latest/index.html)
- [PyPi](https://pypi.org/project/lassobbn/)

# Additional APIs

turing_bbn                                                                            |  pyspark-bbn
:------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:
![turing_bbn logo](https://turing-bbn.oneoffcoder.com/_images/turing-bbn-150x150.png) |![pyspark-bbn logo](https://pyspark-bbn.oneoffcoder.com/_images/pyspark-bbn-150x150.png)

* [turing_bbn](https://turing-bbn.oneoffcoder.com/) is a C++17 implementation of py-bbn; take your causal and probabilistic inferences to the next computing level!
* [pyspark-bbn](https://pyspark-bbn.oneoffcoder.com/) is a is a scalable, massively parallel processing MPP framework for learning structures and parameters of Bayesian Belief Networks BBNs using [Apache Spark](https://spark.apache.org/).

# Citation

```
@misc{alemi_2021,
title={lasso-bbn},
url={https://lasso-bbn.readthedocs.io/},
author={F. Alemi, J. Vang},
year={2021},
month={Aug}}
```

# Copyright Stuff

## Software

```
Copyright 2021 Farrokh Alemi and Jee Vang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Art Copyright

Copyright 2021 Daytchia Vang

# Sponsor, Love

- [Patreon](https://www.patreon.com/vangj)
- [GitHub](https://github.com/sponsors/vangj)