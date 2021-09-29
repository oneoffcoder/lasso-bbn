from lassobbn.learn import learn_parameters, learn_structure, to_bbn, to_join_tree
import pandas as pd


# Learn the structure
df_path = './data/data-binary.csv'
meta_path = './data/data-binary-complete.json'

parents = learn_structure(df_path, meta_path, n_way=2, ignore_neg_gt=-0.01, ignore_pos_lt=0.05)
print('parents')
print(parents)
print('-' * 15)

# Learn the parameters
d, g, p = learn_parameters(df_path, parents)
print('domains')
print(d)
print('-' * 15)

print('structure')
for pa, ch in g.edges():
    print(f'{pa} -> {ch}')
print('-' * 15)

print('parameters')
for k, arr in p.items():
    probs = [f'{v:.2f}' for v in arr]
    probs = ', '.join(probs)
    print(f'{k}: [{probs}]')
print('-' * 15)

# Get the BBN
bbn = to_bbn(d, g, p)
jt = to_join_tree(bbn)

print('bbn')
print(bbn)
print('-' * 15)

print('join tree')
print(jt)
print('-' * 15)

# get posteriors
print('posteriors')
mdf = pd.DataFrame([{**{'name': node}, **{val: prob for val, prob in posteriors.items()}}
                    for node, posteriors in jt.get_posteriors().items()])
mdf.index = mdf['name']
mdf = mdf.drop(columns=['name'])
print(mdf)
