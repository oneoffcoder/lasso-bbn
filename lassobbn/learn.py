import json
import operator
from functools import reduce
from itertools import combinations, chain
from typing import List, Tuple, Dict, Union

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.dag import is_directed_acyclic_graph
from pybbn.graph.dag import Bbn
from pybbn.graph.jointree import JoinTree
from pybbn.pptc.inferencecontroller import InferenceController
from sklearn.linear_model import LogisticRegression


def get_ordering_map(meta: Dict[str, any]) -> Dict[str, List[str]]:
    """
    Gets a dictionary specifying ordering. A key is a variable, a value
    is a list of variables that comes before.

    :param meta: Metadata.
    :return: Ordering.
    """
    ordering_map = {}

    col_ordering = list(reversed(meta['ordering']))
    for i, arr in enumerate(col_ordering):
        for col in arr:
            indeps = list(chain(*col_ordering[i + 1:]))
            ordering_map[col] = indeps
    return ordering_map


def get_start_nodes(meta: Dict[str, any]) -> List[str]:
    """
    Gets a list of start variables/nodes to kick off the algorithm.

    :param meta: Metadata.
    :return: Start nodes.
    """
    ordering = meta['ordering']
    return ordering[-1]


def extract_meta(meta_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extracts meta data.
    :param meta_path: Metadata path (JSON file).
    :return: Tuple; (ordering map, start nodes).
    """
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    ordering_map = get_ordering_map(meta)
    start_nodes = get_start_nodes(meta)
    return ordering_map, start_nodes


def get_n_way(X_cols: List[str], n_way=3) -> List[Tuple[str, ...]]:
    """
    Gets up to all n-way interactions.

    :param X_cols: List of variables.
    :param n_way: Maximum n-way interactions. Default is ``3``.
    :return: List of n-way interactions.
    """
    combs = (combinations(X_cols, n + 1) for n in range(n_way))
    combs = chain(*combs)
    combs = list(combs)
    return combs


def get_data(df_path: str, X_cols: List[str], y_col: str, n_way=3) -> pd.DataFrame:
    """
    Gets a data frame with additional columns representing the n-way interactions.

    :param df_path: Path to CSV file.
    :param X_cols: List of variables.
    :param y_col: The dependent variable.
    :param n_way: Number of n-way interactions. Default is ``3``.
    :return: Data frame.
    """

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
                  C=0.2) -> LogisticRegression:
    """
    Performs regression.

    :param X_cols: Independent variables.
    :param y_col: Dependent variable.
    :param df: Data frame.
    :param solver: Solver. Default is liblinear.
    :param penalty: Penalty. Default is ``l1``.
    :param C: Strength of regularlization. Default is ``0.2``.
    :return: Logistic regression model.
    """
    X = df[X_cols]
    y = df[y_col]

    model = LogisticRegression(penalty=penalty, solver=solver, C=C)
    model.fit(X, y)

    return model


def extract_model_params(independent_cols: List[str], y_col: str, model: LogisticRegression) -> \
        Dict[str, Union[str, float]]:
    """
    Extracts parameters from models (e.g. coefficients).

    :param independent_cols: List of independent variables.
    :param y_col: Dependent variable.
    :param model: Logistic regression model.
    :return: Parameters (e.g. coefficients of each independent variable).
    """
    intercept = {'__intercept': model.intercept_[0]}
    indeps = {c: v for c, v in zip(independent_cols, model.coef_[0])}
    y = {'__dependent': y_col}

    d = {**y, **intercept}
    d = {**d, **indeps}

    return d


def to_robustness_indication(params: pd.DataFrame, ignore_neg_gt=-0.1, ignore_pos_lt=0.1) -> pd.DataFrame:
    """
    Checks if each coefficient value is "robust". A coefficient is NOT robust
    if it is less ``ignore_neg_gt`` or if it is less than ``ignore_pos_lt``.

    :param params: Data frame of parameters.
    :param ignore_neg_gt: Threshold. Default is ``-0.1``.
    :param ignore_pos_lt: Threshold. Default is ``0.1``.
    :return: Data frame (all 1's and 0's) indicating robustness.
    """

    def is_robust(v):
        if v < ignore_neg_gt:
            return 0
        if v < ignore_pos_lt:
            return 0
        return 1

    return params[[c for c in params if c not in ['__intercept', '__dependent']]].applymap(is_robust)


def get_robust_stats(robust: pd.DataFrame, robust_threshold=0.9) -> pd.DataFrame:
    """
    Computes the robustness statistics.

    :param robust: Data frame of robustness indicators.
    :param robust_threshold: Threshold for robustness. Default is ``0.9``.
    :return: Data frame of variables that are robust.
    """
    s = robust.sum()
    p = s / robust.shape[0]
    i = s.index

    df = pd.DataFrame([{'name': name, 'count': count, 'percent': pct} for name, count, pct in zip(i, s, p)])
    df = df.sort_values(['count', 'percent', 'name'], ascending=[False, False, True])
    df = df[df['percent'] >= robust_threshold]
    return df


def do_robust_regression(X_cols: List[str], y_col: str, df_path: str, n_way=3,
                         ignore_neg_gt=-0.1, ignore_pos_lt=0.1,
                         n_regressions=10, solver='liblinear', penalty='l1', C=0.2,
                         robust_threshold=0.9) -> Dict[str, Union[str, List]]:
    """
    Performs robust regression.

    :param X_cols: List of independent variables.
    :param y_col: Dependent variable.
    :param df_path: Path of CSV file.
    :param n_way: Number of n-way interactions. Default is 3.
    :param ignore_neg_gt: Threshold for ignoring negative coefficients.
    :param ignore_pos_lt: Threshold for ignoring positive coefficients.
    :param n_regressions: The number of regressions to do. Default is 10.
    :param solver: Solver. Default is ``liblinear``.
    :param penalty: Penalty. Default is ``l1``.
    :param C: Regularization strength. Default is ``0.2``.
    :param robust_threshold: Robustness threshold. Default is ``0.9``.
    :return: A dictionary storing parents of a child. The parents are said to be robust.
    """
    data = get_data(df_path, X_cols, y_col, n_way=n_way)
    frames = (data.sample(frac=0.9) for _ in range(n_regressions))

    independent_cols = [c for c in data.columns if c != y_col]
    models = (do_regression(independent_cols, y_col, data, solver=solver, penalty=penalty, C=C) for df in frames)

    params = pd.DataFrame((extract_model_params(independent_cols, y_col, m) for m in models))
    robust = to_robustness_indication(params, ignore_neg_gt, ignore_pos_lt)
    robust_stats = get_robust_stats(robust, robust_threshold=robust_threshold)

    relationships = {
        'child': y_col,
        'parents': list(robust_stats['name'])
    }

    return relationships


def do_learn(df_path: str, nodes: List[str], seen: Dict[str, List[str]], ordering_map: Dict[str, List[str]], n_way=3,
             ignore_neg_gt=-0.1, ignore_pos_lt=0.1,
             n_regressions=10, solver='liblinear', penalty='l1', C=0.2,
             robust_threshold=0.9) -> None:
    """
    Recursively learns parents or robust independent variables associated with
    each variable.

    :param df_path: CSV path.
    :param nodes: List of variables.
    :param seen: Dictionary storing processed/seen variables.
    :param ordering_map: Ordering map.
    :param n_way: Number of n-way interactions. Default is 3.
    :param ignore_neg_gt: Threshold for ignoring negative coefficients.
    :param ignore_pos_lt: Threshold for ignoring positive coefficients.
    :param n_regressions: The number of regressions to do. Default is 10.
    :param solver: Solver. Default is ``liblinear``.
    :param penalty: Penalty. Default is ``l1``.
    :param C: Regularization strength. Default is ``0.2``.
    :param robust_threshold: Robustness threshold. Default is ``0.9``.
    :return: None.
    """
    next_nodes = []

    for y_col in nodes:
        if y_col in seen:
            continue

        rels = do_robust_regression(ordering_map[y_col], y_col, df_path, n_way, ignore_neg_gt, ignore_pos_lt,
                                    n_regressions, solver, penalty, C, robust_threshold)
        seen[y_col] = rels['parents']
        print(f'{len(seen)} / {len(ordering_map)} | {y_col}')

        component_parents = list(set(chain(*[pa.split('!') for pa in rels['parents']])))
        next_nodes.extend(component_parents)

    next_nodes = list(set(next_nodes))
    next_nodes = [n for n in next_nodes if n not in seen]
    next_nodes = [n for n in next_nodes if len(ordering_map[n]) > 0]

    if len(next_nodes) > 0:
        do_learn(df_path, next_nodes, seen, ordering_map, n_way, ignore_neg_gt, ignore_pos_lt, n_regressions, solver,
                 penalty, C, robust_threshold)


def learn_structure(df_path: str, meta_path: str, n_way=3,
                    ignore_neg_gt=-0.1, ignore_pos_lt=0.1,
                    n_regressions=10, solver='liblinear', penalty='l1', C=0.2,
                    robust_threshold=0.9) -> Dict[str, List[str]]:
    """
    Kicks off the learning process.

    :param df_path: CSV path.
    :param meta_path: Metadata path.
    :param n_way: Number of n-way interactions. Default is 3.
    :param ignore_neg_gt: Threshold for ignoring negative coefficients.
    :param ignore_pos_lt: Threshold for ignoring positive coefficients.
    :param n_regressions: The number of regressions to do. Default is 10.
    :param solver: Solver. Default is ``liblinear``.
    :param penalty: Penalty. Default is ``l1``.
    :param C: Regularization strength. Default is ``0.2``.
    :param robust_threshold: Robustness threshold. Default is ``0.9``.
    :return: Dictionary where keys are children and values are list of parents.
    """
    ordering_map, nodes = extract_meta(meta_path)
    seen = {}
    do_learn(df_path, nodes, seen, ordering_map, n_way, ignore_neg_gt, ignore_pos_lt, n_regressions, solver, penalty, C,
             robust_threshold)
    return trim_relationships(seen)


def trim_parents(parents: List[str]) -> List[str]:
    """
    Prunes or trims down the list of parents. There might be duplicates as a
    result of compound or n-way interactions.

    :param parents: List of parents.
    :return: List of (pruned/trimmed) parents.
    """

    def is_contained_within(pa, pa_sets):
        for s in pa_sets:
            if pa in s:
                return True
        return False

    intera_pas = [set(pa.split('!')) for pa in parents if pa.find('!') != -1]
    single_pas = [pa for pa in parents if pa.find('!') < 0 and is_contained_within(pa, intera_pas) is False]
    pas = single_pas + [pa for pa in parents if pa.find('!') != -1]
    return pas


def trim_relationships(rels: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Trims/prune parent-child relationships.

    :param rels: Dictionary of parent-child relationships.
    :return: Dictionary of trimmed parent-child relationships.
    """
    return {k: trim_parents(pas) for k, pas in rels.items() if len(pas) > 0}


def expand_data(df_path: str, parents: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Expands data with additional columns defined by parent-child relationships.

    :param df_path: CSV path.
    :param parents: Parent-child relationships.
    :return: Data frame.
    """

    def get_interactions(values):
        interactions = sorted(list(set(values)))
        interactions = filter(lambda s: s.find('!') > 0, interactions)
        interactions = map(lambda s: (s, s.split('!')), interactions)
        interactions = {k: v for k, v in interactions}

        return interactions

    df = pd.read_csv(df_path)

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


def learn_parameters(df_path: str, pas: Dict[str, List[str]]) -> \
        Tuple[Dict[str, List[str]], nx.DiGraph, Dict[str, List[float]]]:
    """
    Gets the parameters.

    :param df_path: CSV file.
    :param pas: Parent-child relationships (structure).
    :return: Tuple; first item is dictionary of domains; second item is a graph; third item is dictionary of probabilities.
    """

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

    df = expand_data(df_path, pas)
    g = get_graph(pas)

    ddf = vals_to_str()
    nodes = list(g.nodes())

    domains = {n: sorted(list(ddf[n].unique())) for n in nodes}
    parents = {ch: list(g.predecessors(ch)) for ch in nodes}

    p = {ch: get_total(get_filters(ch, parents, domains), len(domains[ch])) for ch in nodes}
    return domains, g, p


def get_graph(parents: Dict[str, List[str]]) -> nx.DiGraph:
    """
    Gets a graph ``nx.DiGraph``.

    :param parents: Dictionary; keys are children, values are list of parents.
    :return: Graph.
    """
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


def to_bbn(d: Dict[str, List[str]], s: nx.DiGraph, p: Dict[str, List[float]]) -> Bbn:
    """
    Converts the structure and parameters to a BBN.

    :param d: Domain of each variable.
    :param s: Structure.
    :param p: Parameter.
    :return: BBN.
    """

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
        return [{'pa': pa, 'ch': ch} for pa, ch in s.edges()]

    nodes = {name: get_node(name, n_id) for n_id, name in enumerate(s.nodes())}
    edges = get_edges()

    data = {
        'nodes': nodes,
        'edges': edges
    }

    return Bbn.from_dict(data)


def to_join_tree(bbn: Bbn) -> JoinTree:
    """
    Converts a BBN to a Join Tree.

    :param bbn: BBN.
    :return: Join Tree.
    """
    return InferenceController.apply(bbn)


def posteriors_to_df(jt: JoinTree) -> pd.DataFrame:
    """
    Converts posteriors to data frame.

    :param jt: Join tree.
    :return: Data frame.
    """
    mdf = pd.DataFrame([{**{'name': node}, **{val: prob for val, prob in posteriors.items()}}
                        for node, posteriors in jt.get_posteriors().items()])
    mdf.index = mdf['name']
    mdf = mdf.drop(columns=['name'])
    return mdf
