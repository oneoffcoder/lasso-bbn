# lasso-bbn

Learning Bayesian Belief Networks with LASSO. Example code is as below. 

```python
import pandas as pd
from lassobbn.learn import do_learn

df = pd.read_csv('./path/to/data.csv')
bbn_specs = do_learn(df)

print(bbn_specs)
```