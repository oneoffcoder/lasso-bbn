Data
====

Your data should be a comma-seperated value (CSV) format. All your data should be binary in nature, with the values of ``0`` or ``1``. Here is an example of the CSV data you will need.

.. literalinclude:: code/data-binary.csv
   :linenos:

Noticed that the first line contains the ``headers`` which represent the names of the variables. In this example file, there are 5 variables ``a, b, c, d, e``. Also note there are no missing data. This CSV file should be easily read by ``Pandas`` using ``pd.read_csv(...)``.

Since the structure learning algorithm (currently) requires ordering among the variables, the order of the variables in the CSV file is the specified order given to the algorithm. In this example, ``a`` comes before ``b, c, d, e`` and ``b`` comes before ``c, d, e`` and so on. Make sure you understand that the order of the variables in the CSV file is meaningful, as it specifies which variables may be used to regress on others.

.. admonition:: TODO

   - *D00*: In the future, we will enable other types of variables such as continuous and general categorical variables.