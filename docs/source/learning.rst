Learning
========

Learning a Bayesian Belief Network (BBN) means to learn the structure and parameters. The structure of a BBN is typically learned first, and then the parameters are learned afterwards. The signature of the the ``do_learn(...)`` method is as follows.

.. code-block:: python

   do_learn(df: pd.DataFrame, meta: Dict[Any, Any], solver='liblinear', penalty='l1', C=0.2) -> Dict


Since we are using logistic regression with LASSO regularization, you will need to specify how to accomplish the regression with some arguments. The solver can be either ``liblinear`` or ``saga``. The penalty must be ``l1`` and the regularization strength, ``C`` is a number between [0, 1]. For ``C``, a smaller value means stronger regularlization. Please take a look at Scikit's official `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ for additional information.

What is returned is a Python dictionary that stores the structure and parameters of the learned BBN. Here is an example of the dictionary that is returned.

.. literalinclude:: code/do-learn-result.json
   :language: json
   :linenos:

.. admonition:: TODO

   - *L00*: Implement LASSO regression with continuous dependent variable.
   - *L01*: Implement LASSO regression with categorical independent variable.
   - *L02*: How do we implement LASSO regression with categorical dependent variable?
   - *L03*: How do we learn with partial ordering of the variables?
   - *L04*: How do we learn with no ordering of the variables?