Meta Information
================

Meta information is information that will help guide the learning procedure. The learning procedure will consider the following.

- The ordering of the variables. For now, a complete and partial ordering is allowed.
- A list of edges to blacklist. These are edges that will never be allowed even if they are found.
- A list of edges to whitelist. These are edges that will always be created even if they are not found.

The meta information you provide should be stored in a JSON file format. Below is example of meta information stored in a JSON file that defines complete ordering. Look at the key ``ordering`` and its associated value. The value is a list of lists (sub-lists) or a nested list. The sequence of the variables are stored inside these sub-lists. Here, we have 5 sub-lists, and in each sublist, only a single element. This ordering implies that ``a`` comes before ``b``, ``b`` comes before ``c`` and so on. This ordering is a complete ordering since there are no more than one element in each sub-list. Notice that each sub-list is a level of sequence, where variables in earlier sub-lists occur before those in later ones.

.. literalinclude:: code/data-binary-complete.json
   :language: json
   :linenos:

Take a look at this next ordering. This ordering is a partial ordering since there is at least one sub-list that has more than one element. In particular, this ordering is not complete since we do not now if ``a`` comes before ``b`` or vice-versa; we have incomplete knowledge. Thus, we specify ``a`` and ``b`` to be at the same level of sequence. For variables at the same level of sequence, they will never be considered as dependent variables of one another (since we do not know their ordering). The meaning of this ordering is that ``a`` and ``b`` comes before ``c`` and ``d``, and ``c`` and ``d`` comes before ``e``.

.. literalinclude:: code/data-binary-partial.json
   :language: json
   :linenos: