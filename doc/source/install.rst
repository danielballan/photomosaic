*******
Install
*******

You can install photomosiac using pip or from source.

First verify that you have Python 3.5+.

.. code-block:: bash

   python3 --version

If necessary, install it by your method of choice (apt, Homebrew, conda, etc.).

Installation Using Pip
======================

.. code-block:: bash

   python3 -m pip install -U photomosaic

Or, to include the optional dependencies for the experimental parallelism
features, install:

.. code-block:: bash

   python3 -m pip install -U photomosaic[parallel]

Development Installation
========================

.. code-block:: bash

    git clone https://github.com/danielballan/photomosaic
    cd photomosaic
    pip install -e .

Or, to include the optional dependencies for the experimental parallelism
features, install:

.. code-block:: bash

   python3 -m pip install -U -e .[parallel]

Development
===========

For development, you will also want the dependencies for running the tests and
building the documentation:

.. code-block:: bash

   python3 -m pip install -Ur requirements-dev.txt

To run the tests:

.. code-block:: bash

   pytest

These ``pytest`` arguments are commonly useful:

* ``-v`` verbose
* ``--lf`` Run only those tests which failed on the last run.
* ``-s`` Do not capture stdout/err per test.
* ``-k EXPRESSION`` Filter tests by pattern-matching test name.

To build the documentation:

.. code-block:: bash

    make -C doc pools  # Generate tile pools used in documentation.
    make -C doc images  # Generate example images used in documentation.
    make -C doc html  # Build the documentation.
