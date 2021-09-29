from lassobbn.learn import learn_parameters, learn_structure, to_bbn, to_join_tree, posteriors_to_df

# Step 1. Learn the structure
df_path = './data/data-binary.csv'
meta_path = './data/data-binary-complete.json'

parents = learn_structure(df_path, meta_path, n_way=2, ignore_neg_gt=-0.01, ignore_pos_lt=0.05)
print('parents')
print(parents)
print('-' * 15)
# {'e': ['d!b'], 'd': ['b!a']}

# Step 2. Learn the parameters
d, g, p = learn_parameters(df_path, parents)
print('domains')
print(d)
print('-' * 15)
# {'d!b': ['0', '1'], 'e': ['0', '1'], 'd': ['0', '1'], 'b': ['0', '1'], 'b!a': ['0', '1'], 'a': ['0', '1']}

print('structure')
for pa, ch in g.edges():
    print(f'{pa} -> {ch}')
print('-' * 15)
# d!b -> e
# d -> d!b
# b -> d!b
# b -> b!a
# b!a -> d
# a -> b!a

print('parameters')
for k, arr in p.items():
    probs = [f'{v:.2f}' for v in arr]
    probs = ', '.join(probs)
    print(f'{k}: [{probs}]')
print('-' * 15)
# d!b: [1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 1.00]
# e: [0.77, 0.23, 0.08, 0.92]
# d: [0.79, 0.21, 0.80, 0.20]
# b: [0.80, 0.20]
# b!a: [1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 1.00]
# a: [0.19, 0.81]

# Step 3. Get the BBN
bbn = to_bbn(d, g, p)

# Step 4. Get the Join Tree
jt = to_join_tree(bbn)

print('bbn')
print(bbn)
print('-' * 15)
# 0|d!b|0,1
# 1|e|0,1
# 2|d|0,1
# 3|b|0,1
# 4|b!a|0,1
# 5|a|0,1
# 0->1
# 2->0
# 3->0
# 3->4
# 4->2
# 5->4

print('join tree')
print(jt)
print('-' * 15)
# (d!b,e)
# (b,d,d!b)
# (b,b!a,d)
# (a,b,b!a)
# |(b,d,d!b) -- d,b -- (b,b!a,d)|
# |(b,b!a,d) -- b,b!a -- (a,b,b!a)|
# |(d!b,e) -- d!b -- (b,d,d!b)|
# (b,d,d!b)--|(b,d,d!b) -- d,b -- (b,b!a,d)|--(b,b!a,d)
# (b,b!a,d)--|(b,b!a,d) -- b,b!a -- (a,b,b!a)|--(a,b,b!a)
# (d!b,e)--|(d!b,e) -- d!b -- (b,d,d!b)|--(b,d,d!b)

# Get posteriors
print('posteriors')
mdf = posteriors_to_df(jt)
print(mdf)

# should print
#              0         1
# name
# d!b   0.960997  0.039003
# e     0.740779  0.259221
# d     0.795200  0.204800
# b     0.802900  0.197100
# b!a   0.840211  0.159789
# a     0.189300  0.810700
