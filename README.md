![pybbn logo](https://lasso-bbn.readthedocs.io/en/latest/_images/logo-250x250.png)

# LASSO BBN

Learning Bayesian Belief Networks (BBNs) with LASSO. Example code is as below. 

```python
import pandas as pd
from lassobbn.learn import do_learn

df = pd.read_csv('./path/to/data.csv')
bbn_specs = do_learn(df)

print(bbn_specs)
```

You can then use [Py-BBN](https://py-bbn.readthedocs.io/) to create a BBN and join tree (JT) instance and perform exact inference.
