Installation
============

You can install featomic in different ways depending on which language you plan
to use it from.

.. _install-python-lib:

Installing the Python module
----------------------------

Pre-compiled wheels
^^^^^^^^^^^^^^^^^^^

The easiest way to install featomic is to use `pip <https://pip.pypa.io>`_.

.. code-block:: bash

    pip install --upgrade pip
    pip install featomic


Building from source
^^^^^^^^^^^^^^^^^^^^

If you want to build the code from source, you'll need a Rust compiler, which
you can install using `rustup <https://rustup.rs/>`_ or your OS package manager;
and `git <https://git-scm.com>`_.

.. code-block:: bash

    # Make sure you are using the latest version of pip
    pip install --upgrade pip

    git clone https://github.com/metatensor/featomic
    cd featomic
    pip install .

    # alternatively, the same thing in a single command
    pip install git+https://github.com/metatensor/featomic


.. _install-c-lib:

Installing the C/C++ library
----------------------------

This installs a C-compatible shared library that can also be called from C++, as
well as CMake files that can be used with ``find_package(featomic)``.

.. code-block:: bash

    git clone https://github.com/metatensor/featomic
    cd featomic/featomic
    mkdir build
    cd build
    cmake <CMAKE_OPTIONS_HERE> ..
    make install

The build and installation can be configures with a few cmake options, using
``-D<OPTION>=<VALUE>`` on the cmake command line, or one of the cmake GUI
(``cmake-gui`` or ``ccmake``). Here are the main configuration options:

+--------------------------------------+-----------------------------------------------+----------------+
| Option                               | Description                                   | Default        |
+======================================+===============================================+================+
| CMAKE_BUILD_TYPE                     | Type of build: debug or release               | release        |
+--------------------------------------+-----------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX                 | Prefix in which the library will be installed | ``/usr/local`` |
+--------------------------------------+-----------------------------------------------+----------------+
| INCLUDE_INSTALL_DIR                  | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``include``    |
|                                      |  where the headers will be installed          |                |
+--------------------------------------+-----------------------------------------------+----------------+
| LIB_INSTALL_DIR                      | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``lib``        |
|                                      | where the shared library will be installed    |                |
+--------------------------------------+-----------------------------------------------+----------------+
| BUILD_SHARED_LIBS                    | Default to installing and using a shared      | ON             |
|                                      | library instead of a static one               |                |
+--------------------------------------+-----------------------------------------------+----------------+
| FEATOMIC_INSTALL_BOTH_STATIC_SHARED  | Install both the shared and static version    | ON             |
|                                      | of the library                                |                |
+--------------------------------------+-----------------------------------------------+----------------+
| FEATOMIC_FETCH_METATENSOR            | Automatically fetch, build and install        | OFF            |
|                                      | metatensor (a dependency of featomic)         |                |
+--------------------------------------+-----------------------------------------------+----------------+

Using the Rust library
----------------------

Add the following to your project ``Cargo.toml``

.. code-block:: toml

    [dependencies]
    featomic = {git = "https://github.com/metatensor/featomic"}


.. _install-torch-script:

Installing the TorchScript bindings
-----------------------------------

For usage from Python
^^^^^^^^^^^^^^^^^^^^^

You can install the code with ``pip``:

.. code-block:: bash

    pip install --upgrade pip
    pip install featomic[torch]


You can also build the code from source

.. code-block:: bash

    pip install --upgrade pip

    git clone https://github.com/metatensor/featomic
    cd featomic/python/featomic_torch
    pip install .

    # alternatively, the same thing in a single command
    pip install git+https://github.com/metatensor/featomic#subdirectory=python/featomic_torch


For usage from C++
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/metatensor/featomic
    cd featomic/featomic-torch
    mkdir build && cd build
    cmake ..
    # configure cmake if needed
    cmake --build . --target install

Compiling the TorchScript bindings requires you to manually install some of the
dependencies:

- the C++ part of PyTorch, which you can install `on it's own
  <https://pytorch.org/get-started/locally/>`_. You can also use the
  installation that comes with a Python installation by adding the output of the
  command below to ``CMAKE_PREFIX_PATH``:

  .. code-block:: bash

    python -c "import torch; print(torch.utils.cmake_prefix_path)"

- :ref:`the C++ interface of featomic <install-c-lib>`, which itself requires
  the `C++ interface of metatensor`_;
- the `TorchScript interface of metatensor`_. We can download and build an
  appropriate version of it automatically by setting the cmake option
  ``-DFEATOMIC_FETCH_METATENSOR_TORCH=ON``
- the `TorchScript interface of metatomic`_. We can download and build an
  appropriate version of it automatically by setting the cmake option
  ``-DFEATOMIC_FETCH_METATOMIC_TORCH=ON``

If any of these dependencies is not in a standard location, you should specify
the installation directory when configuring cmake with ``CMAKE_PREFIX_PATH``.
Other useful configuration options are:

+----------------------------------+-----------------------------------------------+----------------+
| Option                           | Description                                   | Default        |
+==================================+===============================================+================+
| CMAKE_BUILD_TYPE                 | Type of build: debug or release               | release        |
+----------------------------------+-----------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX             | Prefix in which the library will be installed | ``/usr/local`` |
+----------------------------------+-----------------------------------------------+----------------+
| CMAKE_PREFIX_PATH                | ``;``-separated list of path where CMake will |                |
|                                  | search for dependencies.                      |                |
+----------------------------------+-----------------------------------------------+----------------+
| FEATOMIC_FETCH_METATENSOR_TORCH  | Should CMake automatically download and       | OFF            |
|                                  | install metatensor-torch?                     |                |
+----------------------------------+-----------------------------------------------+----------------+
| FEATOMIC_FETCH_METATOMIC_TORCH   | Should CMake automatically download and       | OFF            |
|                                  | install metatomic-torch?                      |                |
+----------------------------------+-----------------------------------------------+----------------+

.. _C++ interface of metatensor: https://docs.metatensor.org/latest/installation.html#install-c
.. _TorchScript interface of metatensor: https://docs.metatensor.org/latest/installation.html#install-torch-cxx
.. _TorchScript interface of metatomic: https://docs.metatensor.org/metatomic/latest/installation.html#install-torch-cxx
