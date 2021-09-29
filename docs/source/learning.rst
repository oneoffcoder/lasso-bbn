Learning
========

Learning a Bayesian Belief Network (BBN) means to learn the structure and parameters. The structure of a BBN is typically learned first, and then the parameters are learned afterwards. The signature of the the ``learn_structure(...)`` method is as follows.

.. code-block:: python

   learn_structure(df_path: str, meta_path: str, n_way=3,
                    ignore_neg_gt=-0.1, ignore_pos_lt=0.1,
                    n_regressions=10, solver='liblinear', penalty='l1', C=0.2,
                    robust_threshold=0.9) -> Dict[str, List[str]]


Since we are using logistic regression with LASSO regularization, you will need to specify how to accomplish the regression with some arguments. The solver can be either ``liblinear`` or ``saga``. The penalty must be ``l1`` and the regularization strength, ``C`` is a number between [0, 1]. For ``C``, a smaller value means stronger regularlization. Please take a look at Scikit's official `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ for additional information.

What is returned is a Python dictionary that stores the child to parent relationships. Here is an example of the dictionary that is returned.

.. literalinclude:: code/learn-structure.json
   :language: json
   :linenos:

After you have learned the child to parent relationships (or equivalently, parent to child relationships), you should then learn the parameters. The signature of the ``learn_parameters(...)`` function is as follows.

.. code-block:: python

    learn_parameters(df_path: str, pas: Dict[str, List[str]]) -> \
        Tuple[Dict[str, List[str]], nx.DiGraph, Dict[str, List[float]]]

The output of ``learn_parameters(...)`` is a tuple of 3 things.

- domains of each variable
- graphical structure
- conditional probability tables for each variable



.. admonition:: TODO

   - *L00*: Implement LASSO regression with continuous dependent variable.
   - *L01*: Implement LASSO regression with categorical independent variable.
   - *L02*: How do we implement LASSO regression with categorical dependent variable?
   - *L03*: How do we learn with partial ordering of the variables? (DONE)
   - *L04*: How do we learn with no ordering of the variables?
   - *L05*: Implement blacklisted or whitelisted edges.