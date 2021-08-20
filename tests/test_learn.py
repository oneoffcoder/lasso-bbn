import json

import pandas as pd

from lassobbn.learn import do_learn, to_bbn


def test_learn_complete_ordering():
    """
    Tests learning structure and parameters with complete ordering.
    :return: None.
    """
    df = pd.read_csv('./data/data-binary.csv')
    with open('./data/data-binary-complete.json', 'r') as f:
        meta = json.load(f)
    observed = do_learn(df, meta)
    expected = {'nodes': {'a': {'probs': [0.1893, 0.8107], 'variable': {'id': 0, 'name': 'a', 'values': ['0', '1']}},
                          'b': {'probs': [0.8029, 0.1971], 'variable': {'id': 1, 'name': 'b', 'values': ['0', '1']}},
                          'c': {
                              'probs': [0.8085526315789474, 0.19144736842105264, 0.2225201072386059, 0.7774798927613941,
                                        0.8010447073283147, 0.19895529267168535, 0.2146433041301627,
                                        0.7853566958698373], 'variable': {'id': 2, 'name': 'c', 'values': ['0', '1']}},
                          'd': {
                              'probs': [0.7703947368421052, 0.22960526315789473, 0.8257372654155496, 0.1742627345844504,
                                        0.7970502381318175, 0.2029497618681825, 0.8041301627033792,
                                        0.19586983729662077], 'variable': {'id': 3, 'name': 'd', 'values': ['0', '1']}},
                          'e': {
                              'probs': [0.9053310268910206, 0.0946689731089794, 0.11197604790419162, 0.8880239520958084,
                                        0.903954802259887, 0.096045197740113, 0.08465608465608465, 0.9153439153439153],
                              'variable': {'id': 4, 'name': 'e', 'values': ['0', '1']}}},
                'edges': [{'pa': 'a', 'ch': 'c'}, {'pa': 'a', 'ch': 'd'}, {'pa': 'b', 'ch': 'c'},
                          {'pa': 'b', 'ch': 'd'}, {'pa': 'b', 'ch': 'e'}, {'pa': 'd', 'ch': 'e'}]}
    assert observed == expected


def test_learn_partial_ordering():
    """
    Tests learning structure and parameters with partial ordering.
    :return: None.
    """
    df = pd.read_csv('./data/data-binary.csv')
    with open('./data/data-binary-partial.json', 'r') as f:
        meta = json.load(f)
    observed = do_learn(df, meta)
    expected = {'nodes': {'a': {'probs': [0.1893, 0.8107], 'variable': {'id': 0, 'name': 'a', 'values': ['0', '1']}},
                          'b': {'probs': [0.8029, 0.1971], 'variable': {'id': 1, 'name': 'b', 'values': ['0', '1']}},
                          'c': {
                              'probs': [0.8085526315789474, 0.19144736842105264, 0.2225201072386059, 0.7774798927613941,
                                        0.8010447073283147, 0.19895529267168535, 0.2146433041301627,
                                        0.7853566958698373], 'variable': {'id': 2, 'name': 'c', 'values': ['0', '1']}},
                          'd': {
                              'probs': [0.7703947368421052, 0.22960526315789473, 0.8257372654155496, 0.1742627345844504,
                                        0.7970502381318175, 0.2029497618681825, 0.8041301627033792,
                                        0.19586983729662077], 'variable': {'id': 3, 'name': 'd', 'values': ['0', '1']}},
                          'e': {
                              'probs': [0.9053310268910206, 0.0946689731089794, 0.11197604790419162, 0.8880239520958084,
                                        0.903954802259887, 0.096045197740113, 0.08465608465608465, 0.9153439153439153],
                              'variable': {'id': 4, 'name': 'e', 'values': ['0', '1']}}},
                'edges': [{'pa': 'a', 'ch': 'c'}, {'pa': 'a', 'ch': 'd'}, {'pa': 'b', 'ch': 'c'},
                          {'pa': 'b', 'ch': 'd'}, {'pa': 'b', 'ch': 'e'}, {'pa': 'd', 'ch': 'e'}]}

    assert observed == expected


def test_to_bbn_with_complete_ordering():
    """
    Test converting structure/parameter dictionary to BBN with complete ordering.
    :return: None.
    """
    df = pd.read_csv('./data/data-binary.csv')
    with open('./data/data-binary-complete.json', 'r') as f:
        meta = json.load(f)
    json_data = do_learn(df, meta)
    bbn = to_bbn(json_data)

    observed = str(bbn).split('\n')
    expected = '''0|a|0,1
        1|b|0,1
        2|c|0,1
        3|d|0,1
        4|e|0,1
        0->2
        0->3
        1->2
        1->3
        1->4
        3->4'''.strip().split('\n')

    observed = [o.strip() for o in observed]
    expected = [e.strip() for e in expected]

    assert bbn is not None
    assert observed == expected


def test_to_bbn_with_partial_ordering():
    """
    Test converting structure/parameter dictionary to BBN with partial ordering.
    :return: None.
    """
    df = pd.read_csv('./data/data-binary.csv')
    with open('./data/data-binary-partial.json', 'r') as f:
        meta = json.load(f)
    json_data = do_learn(df, meta)
    bbn = to_bbn(json_data)

    observed = str(bbn).split('\n')
    expected = '''0|a|0,1
        1|b|0,1
        2|c|0,1
        3|d|0,1
        4|e|0,1
        0->2
        0->3
        1->2
        1->3
        1->4
        3->4'''.strip().split('\n')

    observed = [o.strip() for o in observed]
    expected = [e.strip() for e in expected]

    assert bbn is not None
    assert observed == expected
