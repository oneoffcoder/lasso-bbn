from lassobbn.learn import learn_parameters, learn_structure, to_bbn, to_join_tree

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
print(g.edges())
print('-' * 15)

print('parameters')
print(p)
print('-' * 15)

# Get the BBN
bbn = to_bbn(d, g, p)
jt = to_join_tree(bbn)

print('bbn')
print(bbn)
print('-' * 15)

print('join tree')
print(jt)
