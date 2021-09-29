Inference
=========

After you learned the structure and parameters of the BBN, then you can use `Py-BBN <https://py-bbn.readthedocs.io/>`_ to perform inference. First, you have to create an instance of a BBN, and then use that BBN instance to create an instance of a Junction Tree (JT). Py-BBN is opened source may be installed on `PyPi <https://pypi.org/project/pybbn/>`_. This library already lists Py-BBN as a requirement, and by installing this library, you will also install Py-BBN. The methods that you need to pay attention to are as follows.

- ``to_bbn(d, g, p)`` : uses the domain information ``d``, structure ``g`` and parameters ``p`` to create a Bayesian Belief Network (BBN)
- ``to_join_tree(bbn)`` : converts a BBN to a Join Tree (JT)
- ``posters_to_df(jt)`` : gets the posterior information as a data frame

.. literalinclude:: code/demo.py
   :language: python
   :linenos:
   :lines: 44-