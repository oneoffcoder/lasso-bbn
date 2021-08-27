import json
import operator
from functools import reduce
from itertools import combinations, chain
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
import numpy as np


def do_robust_regression(X_cols: List[str], y_col: str, df_path: str, n_way=3,
                         ignore_neg_gt=-0.1, ignore_pos_lt=0.1,
                         n_regressions=10, solver='liblinear', penalty='l1', C=0.2,
                         robust_threshold=0.9):
    def get_n_way(X_cols: List[str], n_way=3):
        combs = (combinations(X_cols, n + 1) for n in range(n_way))
        combs = chain(*combs)
        combs = list(combs)
        return combs

    def get_data(df_path: str, X_cols: List[str], y_col: str, n_way=3):
        def to_col_name(interaction):
            if len(interaction) == 1:
                return interaction[0]
            else:
                return '!'.join(interaction)

        def get_interaction(interaction):
            def multiply(r):
                vals = [r[col] for col in interaction]
                return reduce(operator.mul, vals, 1)

            return data.apply(multiply, axis=1)

        data = pd.read_csv(df_path)
        interactions = get_n_way(X_cols, n_way=n_way)

        d = {to_col_name(interaction): get_interaction(interaction) for interaction in interactions}
        d = {**d, **{y_col: data[y_col]}}

        df = pd.DataFrame(d)
        return df

    def do_regression(X_cols: List[str], y_col: str, df: pd.DataFrame, solver='liblinear', penalty='l1',
                      C=0.2) -> pd.DataFrame:
        X = df[X_cols]
        y = df[y_col]

        model = LogisticRegression(penalty=penalty, solver=solver, C=C)
        model.fit(X, y)

        return model

    def extract_model_params(independent_cols: List[str], y_col: str, model: LogisticRegression):
        intercept = {'__intercept': model.intercept_[0]}
        indeps = {c: v for c, v in zip(independent_cols, model.coef_[0])}
        y = {'__dependent': y_col}

        d = {**y, **intercept}
        d = {**d, **indeps}

        return d

    def to_robustness_indication(params: pd.DataFrame, ignore_neg_gt=-0.1, ignore_pos_lt=0.1):
        def is_robust(v):
            if v < ignore_neg_gt:
                return 0
            if v < ignore_pos_lt:
                return 0
            return 1

        return params[[c for c in params if c not in ['__intercept', '__dependent']]].applymap(is_robust)

    def get_robust_stats(robust: pd.DataFrame, robust_threshold=0.9):
        s = robust.sum()
        p = s / robust.shape[0]
        i = s.index

        df = pd.DataFrame([{'name': name, 'count': count, 'percent': pct} for name, count, pct in zip(i, s, p)])
        df = df.sort_values(['count', 'percent', 'name'], ascending=[False, False, True])
        df = df[df['percent'] >= robust_threshold]
        return df

    data = get_data(df_path, X_cols, y_col, n_way=n_way)
    frames = (data.sample(frac=0.9) for _ in range(n_regressions))

    independent_cols = [c for c in data.columns if c != y_col]
    models = (do_regression(independent_cols, y_col, data, solver=solver, penalty=penalty, C=C) for df in frames)

    params = pd.DataFrame((extract_model_params(independent_cols, y_col, m) for m in models))
    robust = to_robustness_indication(params, ignore_neg_gt, ignore_pos_lt)
    robust_stats = get_robust_stats(robust)

    relationships = {
        'child': y_col,
        'parents': list(robust_stats['name'])
    }

    return relationships


def learn_parents(df_path: str, meta_path: str, n_way=3,
          ignore_neg_gt=-0.1, ignore_pos_lt=0.1,
          n_regressions=10, solver='liblinear', penalty='l1', C=0.2,
          robust_threshold=0.9):
    def get_resources():
        def get_ordering_map(meta):
            ordering_map = {}

            col_ordering = list(reversed(meta['ordering']))
            for i, arr in enumerate(col_ordering):
                for col in arr:
                    indeps = list(chain(*col_ordering[i + 1:]))
                    ordering_map[col] = indeps
            return ordering_map

        def get_start_nodes(meta):
            ordering = meta['ordering']
            return ordering[-1]

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        ordering_map = get_ordering_map(meta)
        start_nodes = get_start_nodes(meta)

        return ordering_map, start_nodes

    def do_learn(nodes, seen, ordering_map):
        next_nodes = []

        for y_col in nodes:
            if y_col in seen:
                continue

            rels = do_robust_regression(ordering_map[y_col], y_col, df_path,
                                        n_way=n_way, ignore_neg_gt=ignore_neg_gt, ignore_pos_lt=ignore_pos_lt,
                                        n_regressions=n_regressions, solver=solver, penalty=penalty,
                                        C=C, robust_threshold=robust_threshold)
            seen[y_col] = rels['parents']
            print(f'{len(seen)} / {len(ordering_map)} | {y_col}')

            component_parents = list(set(chain(*[pa.split('!') for pa in rels['parents']])))
            next_nodes.extend(component_parents)

        next_nodes = list(set(next_nodes))
        next_nodes = [n for n in next_nodes if n not in seen]
        next_nodes = [n for n in next_nodes if len(ordering_map[n]) > 0]

        if len(next_nodes) > 0:
            do_learn(next_nodes, seen, ordering_map)

    ordering_map, nodes = get_resources()
    seen = {}
    do_learn(nodes, seen, ordering_map)
    return seen


def get_structure(rels):
    def trim_parents(parents):
        def is_contained_within(pa, pa_sets):
            for s in pa_sets:
                if pa in s:
                    return True
            return False

        intera_pas = [set(pa.split('!')) for pa in parents if pa.find('!') != -1]
        single_pas = [pa for pa in parents if pa.find('!') < 0 and is_contained_within(pa, intera_pas) is False]
        pas = single_pas + [pa for pa in parents if pa.find('!') != -1]
        return pas

    parents = {k: trim_parents(pas) for k, pas in rels.items() if len(pas) > 0}

    g = nx.DiGraph()

    for ch, pas in parents.items():
        for pa in pas:
            g.add_edge(pa, ch)

            if not is_directed_acyclic_graph(g):
                g.remove_edge(pa, ch)

        for pa in pas:
            pa_set = pa.split('!')
            if len(pa_set) < 2:
                continue

            for single_pa in pa_set:
                g.add_edge(single_pa, pa)

                if not is_directed_acyclic_graph(g):
                    g.remove_edge(single_pa, pa)
    return g


def get_parameters(df: pd.DataFrame, g: nx.DiGraph) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
    """
    Gets the parameters.

    :param df: Data.
    :param g: Graph (structure).
    :return: Tuple; first item is dictionary of domains; second item is dictionary of probabilities.
    """

    def expand_data(df_path: str, pa_path: str):
        def get_interactions(values):
            interactions = sorted(list(set(values)))
            interactions = filter(lambda s: s.find('!') > 0, interactions)
            interactions = map(lambda s: (s, s.split('!')), interactions)
            interactions = {k: v for k, v in interactions}

            return interactions

        df = pd.read_csv(df_path)

        with open(pa_path, 'r') as f:
            parents = json.load(f)

        ch_interactions = get_interactions(chain(*[v for _, v in parents.items()]))
        pa_interactions = get_interactions([k for k, _ in parents.items()])
        interactions = {**ch_interactions, **pa_interactions}

        def expand(r, cols):
            vals = [r[c] for c in cols]
            result = reduce(operator.mul, vals, 1)
            return result

        for col_name, cols in interactions.items():
            df[col_name] = df.apply(lambda r: expand(r, cols), axis=1)

        return df

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
            def is_valid(tups):
                n_tups = len(tups)
                u_tups = len(set([name for name, _ in tups]))
                if n_tups == u_tups:
                    return True
                return False

            vals = [[(pa, v) for v in domains[pa]] for pa in pas]
            vals = vals + [[(ch, v) for v in domains[ch]]]
            vals = chain(*vals)
            vals = combinations(vals, len(pas) + 1)
            vals = filter(is_valid, vals)
            vals = map(lambda tups: ' and '.join([f'`{t[0]}`=="{t[1]}"' for t in tups]), vals)
            vals = list(vals)
            return vals

    def get_total(filters, n):
        def divide(arr):
            a = np.array(arr)
            n = np.sum(a)

            if n == 0:
                p = 1 / len(arr)
                return [p for _ in range(len(arr))]

            r = a / n
            r = list(r)
            return r

        counts = [ddf.query(f).shape[0] for f in filters]
        counts = [counts[i:i + n] for i in range(0, len(counts), n)]
        counts = [divide(arr) for arr in counts]
        counts = list(chain(*counts))
        return counts

    ddf = vals_to_str()
    nodes = list(g.nodes())

    domains = {n: sorted(list(ddf[n].unique())) for n in nodes}
    parents = {ch: list(g.predecessors(ch)) for ch in nodes}

    p = {ch: get_total(get_filters(ch, parents, domains), len(domains[ch])) for ch in nodes}
    return domains, p