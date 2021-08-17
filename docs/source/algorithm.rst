Algorithm
=========

The structure learning algorithm is best understood when a complete ordering of a set of variables is given. Assume we have a causal model for which we know the true structure and parameters, and let's say this model is a causal Bayesian Belief Network (BBN). Let's say this model is shown in the figure below.

Now, let's say we have observed data from this causal BBN, and a sample of the data looks as below.

.. literalinclude:: code/data-binary.csv
   :linenos:

If a user can correctly specify the order of the variables by indicating which variable occurs before which other ones, then we can induce/learn a causal BBN structure from the data. Let's say a user specifies the order to be ``a, b, c, d, e``. Note that even though ``a`` does not come before ``b`` and vice-versa, that is okay, since they are tied and we just need an ordering.

The structure learning algorithm iterates over each variable as a dependent variable while regressing on all those that come before it. Since there are 5 variables, there are a maximum of 5 regression equations to run. Since ``a`` is the first variable and no other variables precede it, we will only run 4 regression equations.

- :math:`b = a`
- :math:`c = a + b`
- :math:`d = a + b + c`
- :math:`e = a + b + c + d`

We will eliminate which independents variables are not a parent of the dependent variable by knowing that the sequence implies time dependency and the coefficient associated with each independent variable indicates prediction strength. Lasso regularization will force the coefficients to zero, and it is expected that each model specified will will have non-zero coefficients for those independent variables that are parents of the dependent variable.

