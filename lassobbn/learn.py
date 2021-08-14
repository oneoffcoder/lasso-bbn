from itertools import chain, combinations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.dag import is_directed_acyclic_graph
from sklearn.linear_model import LogisticRegression


def get_model_params(df, solver='liblinear', penalty='l1', C=0.2):
    def get_model(df, X_cols, y_col):
        X = df[X_cols]
        y = df[y_col]

        model = LogisticRegression(penalty=penalty, solver=solver, C=C)
        model.fit(X, y)

        return model

    def extract_model_params(y, fields, model):
        child = {'child': y}
        intercepts = {'intercept': model.intercept_[0]}
        coefs = {field: coef for field, coef in zip(fields, model.coef_[0])}
        others = {field: 0.0 for field in fields[len(coefs):]}

        p = {**child, **intercepts}
        p = {**p, **coefs}
        p = {**p, **others}

        return p

    args = [(list(df.columns[0:index]), y) for index, y in enumerate(df.columns) if index > 0]
    models = [(y_col, get_model(df, X_cols, y_col)) for X_cols, y_col in args]
    param_df = pd.DataFrame([extract_model_params(y, df.columns, model) for y, model in models])
    return param_df


def get_structure(param_df, threshold=0.0):
    def get_edges(r, nodes):
        edges = []
        ch = r['child']
        for pa in nodes:
            if pa == ch:
                break
            if abs(r[pa]) > threshold:
                edge = (pa, ch)
                edges.append(edge)
        return edges

    nodes = [v for v in param_df.columns if v not in ['child', 'intercept']]
    edges = list(chain(*[get_edges(r, nodes) for _, r in param_df.iterrows()]))

    g = nx.DiGraph()

    for n in nodes:
        g.add_node(n)

    for edge in edges:
        g.add_edge(*edge)
        if not is_directed_acyclic_graph(g):
            g.remove_edge(*edge)

    return g


def get_parameters(df, g):
    def vals_to_str():
        ddf = df.copy(deep=True)
        for col in ddf.columns:
            ddf[col] = ddf[col].astype(str)
        return ddf

    def get_filters(ch, parents, domains):
        pas = parents[ch]
        if len(pas) == 0:
            ch_domain = domains[ch]
            return [f'{ch}=="{v}"' for v in ch_domain]
        else:
            vals = [[(pa, v) for v in domains[pa]] for pa in pas]
            vals = vals + [[(ch, v) for v in domains[ch]]]
            vals = chain(*vals)
            vals = combinations(vals, len(pas) + 1)
            vals = filter(
                lambda tups: tups[0][0] != tups[1][0] and tups[0][0] != tups[2][0] and tups[1][0] != tups[2][0], vals)
            vals = map(lambda tups: ' and '.join([f'{t[0]}=="{t[1]}"' for t in tups]), vals)
            vals = list(vals)
            return vals

    def get_total(filters, n):
        counts = [ddf.query(f).shape[0] for f in filters]
        counts = [counts[i:i + n] for i in range(0, len(counts), n)]
        counts = [list(np.array(arr) / sum(arr)) for arr in counts]
        counts = list(chain(*counts))
        return counts

    ddf = vals_to_str()
    nodes = list(g.nodes())

    domains = {n: sorted(list(ddf[n].unique())) for n in nodes}
    parents = {ch: list(g.predecessors(ch)) for ch in nodes}

    p = {ch: get_total(get_filters(ch, parents, domains), len(domains[ch])) for ch in nodes}
    return domains, p


def do_learn(df, solver='liblinear', penalty='l1', C=0.2):
    def get_node(name, n_id):
        return {
            'probs': p[name],
            'variable': {
                'id': n_id,
                'name': name,
                'values': d[name]
            }
        }

    def get_edges():
        return [{'pa': pa, 'ch': ch} for pa, ch in g.edges()]

    param_df = get_model_params(df, solver=solver, penalty=penalty, C=C)
    g = get_structure(param_df)
    d, p = get_parameters(df, g)

    print(g.nodes())
    print('-' * 15)
    print(g.edges())
    print('-' * 15)
    print(p)
    print('-' * 15)

    json_data = {
        'nodes': {name: get_node(name, n_id) for n_id, name in enumerate(g.nodes())},
        'edges': get_edges()
    }

    print(json_data)
    return json_data
