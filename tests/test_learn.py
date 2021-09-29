from unittest import TestCase

from lassobbn.learn import learn_parameters, learn_structure, to_bbn, to_join_tree


class ApiTest(TestCase):
    def test_process(self):
        df_path = './data/data-binary.csv'
        meta_path = './data/data-binary-complete.json'

        parents = learn_structure(df_path, meta_path, n_way=2, ignore_neg_gt=-0.01, ignore_pos_lt=0.05)
        self.assertDictEqual(parents, {'e': ['d!b'], 'd': ['b!a']})

        # Learn the parameters
        d, g, p = learn_parameters(df_path, parents)
        self.assertDictEqual(d,
                             {'d!b': ['0', '1'], 'e': ['0', '1'], 'd': ['0', '1'], 'b': ['0', '1'], 'b!a': ['0', '1'],
                              'a': ['0', '1']})

        edges = [f'{pa} -> {ch}' for pa, ch in g.edges()]
        self.assertEqual(edges, ['d!b -> e', 'd -> d!b', 'b -> d!b', 'b -> b!a', 'b!a -> d', 'a -> b!a'])

        self.assertDictEqual(p, {'d!b': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                 'e': [0.7674080232799834, 0.23259197672001664, 0.08465608465608465,
                                       0.9153439153439153],
                                 'd': [0.7935015472506546, 0.2064984527493454, 0.8041301627033792, 0.19586983729662077],
                                 'b': [0.8029, 0.1971], 'b!a': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                 'a': [0.1893, 0.8107]}
                             )

        # Get the BBN
        bbn = to_bbn(d, g, p)
        jt = to_join_tree(bbn)

        # get posteriors
        posteriors = [{**{'name': node}, **{val: prob for val, prob in posteriors.items()}}
                      for node, posteriors in jt.get_posteriors().items()]
        self.assertEqual(posteriors, [{'name': 'd!b', '0': 0.960997490478821, '1': 0.03900250952117903},
                                      {'name': 'e', '0': 0.7407789842932012, '1': 0.2592210157067987},
                                      {'name': 'd', '0': 0.7951998827663717, '1': 0.20480011723362845},
                                      {'name': 'b', '0': 0.8029, '1': 0.1971},
                                      {'name': 'b!a', '0': 0.8402110300000001, '1': 0.15978897},
                                      {'name': 'a', '0': 0.18929999999999997, '1': 0.8107}])
