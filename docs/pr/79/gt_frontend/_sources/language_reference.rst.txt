==========================================
Language Reference
==========================================

Abstract
========

This document contains a draft of the DSL syntax for GTScript on meshes.

**Design Goals**

1. Concise, compact and readable syntax
2. General enough to allow generation of efficient code for meshes with unstructured, partially structured, structured meshes
3. Easy to translate into GTScriptAST

Motivation and Scope
====================

For global weather models (e.g. `IFS-FVM <https://refubium.fu-berlin.de/bitstream/handle/fub188/25122/K%C3%BChnlein_FVM_2019.pdf?sequence=1>`__, ICON) the ability to handle non-regular grids is required.

Concepts
========

This section is WIP and will be moved to the concepts repo when it matured.

**Location**

A bounded set with nonempty interior. We differentiate between horizontal locations, i.e. :code:`Cell`, :code:`Edge`, :code:`Vertex` and vertical locations, i.e. a layer. Each location can be attributed to a domain dimesion.

**Mesh**

A subdivision of the computational domain into locations.

**Field**

A mapping between locations and quantities at this location.

.. math::
  f: (c_1, ..., c_n) \mapsto a

**Local field**

A field defined in the neighbourhood of a cell.

**Neighbourhood chain**

Used to select direct or indirect neighbours of a cell. See here https://docs.google.com/document/d/1nQZzKGi7Go9R3fme78ViB_PrQDVPxy4xTzSuGh_yJAI/edit for a description.

Todo: properly define concept and add illustrations.

**Primary location**

The location(s) a stencil is applied to.

**Secondary location**

The location on which operations on neighbours act on.

Stencil
=======

The entry point into GTScript code is a stencil, giving the user a local perspective on the mesh. The syntax to define a stencil follows the regular python syntax to define functions with an additional :code:`@gtscript.stencil` decorator, where the first function argument is the :code:`Mesh` followed by a set of :code:`Field` s.

.. code-block:: python

  @gtscript.stencil
  def stencil_name(mesh: Mesh, my_field : Field[[Vertex], dtype]):
    # ...

**Computations**

The stencils body is composed of one or more computations specifying the iteration domain. The iteration domain is split into horizontal and vertical iteration domains. The vertical iteration domain, which is 1D, can have different policies: FORWARD, BACKWARD and PARALLEL, indicating the order constraints of the iteration. The horizontal iteration domain is of 2-dimensional nature with one unstructured dimension representing the geometrically 2-dimensional horizontal plane. There is only one horizontal iteration policy and it is always PARALLEL, so it is not specified. Statements are executed as specified in the `GTScript parallel model`_

.. _GTScript Parallel Model: https://github.com/GridTools/concepts/wiki/GTScript-Parallel-model

Computations are expressed (syntactically) using :code:`with` statements and the three iteration specifications :code:`computation`, :code:`interval` and :code:`location`.

- :code:`computation(iteration order)`

  Specifies the iteration order in the vertical dimension, where :code:`iteration_policy` is one of: :code:`PARALLEL`, :code:`FORWARD`, :code:`BACKWARD`

- :code:`location(location_type)`

  Restricts the horizontal dimension to locations of type :code:`location_type`.

  .. code-block:: python

    location(Vertex)
    location(Edge)
    location(Cell)

  Locations may be labeled to reference them in neighbor reductions later on as follows:

  .. code-block:: python

    with location(Vertex) as v:
      # ...

- :code:`interval(start, end)`:

  Restricts the vertical dimension to the interval :math:`[start, end[`, i.e. an interval including :code:`start` and excluding :code:`end`.

  .. code-block:: python

    interval(0, 2)      # layer 0 and 1
    interval(0, -1)     # all layers except for the last
    interval(-1, None)  # only the last layer

The skeleton of a stencil :code:`my_stencil` with a single field argument :code:`my_field` defined on vertices, executing concurrently on all vertices, accross all layers then looks as follows:

.. code-block:: python

  @gtscript.stencil
  def my_stencil(mesh: Mesh):
    with computation(PARALLEL), location(Vertex), interval(0, None):
      # ...

The iteration specifications may also be nested as long as their order is :code:`computation`, :code:`location`, :code:`interval`.

.. code-block:: python

  @gtscript.stencil
  def my_stencil(mesh: Mesh):
    with computation(PARALLEL):   # vertical iteration policy
      with location(Vertex):   # location specification
        with interval(0, None):       # vertical interval
          # ...

A stencil running different computations on the first, last and the layers in between could then look as follows:

.. code-block:: python

  @gtscript.stencil
  def my_stencil(mesh: Mesh):
    with computation(PARALLEL):   # vertical iteration policy
      with location(Vertex):   # location specification
        with interval(0, 1):
          # statements executed on the first layer
          # ...
        with interval(1, -1):
          # statements executed on all, but the first and last layer
          # ...
        with interval(-1, None):
          # statements executed on the last layer
          # ...

The specification of the iteration policy and interval may also be skipped in which case the default iteration policy is :code:`PARALLEL` and all layers are considered.

.. code-block:: python

  @gtscript.stencil
  def my_stencil(mesh: Mesh):
    with location(Vertex):
      # ...
    with location(Edge):
      # ...

Todo: Incorporate sparse fields syntax

Types & Variables
-----------------

GTScript only supports the following limited set of types. All variables are required to be of fixed type throughout the stencil.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Type
     - Description
   * - :code:`Field[[*DIMS], V]`
     - A mapping between locations with their attributed dimensions :code:`DIMS` and quantaties of type :code:`V` at this location.
   * - :code:`Mesh`
     - A discrete representation of the computational domain (passed as first argument to the stencil call).
   * - :code:`LocationType`
     - The type of a location, i.e. :code:`Cell`, :code:`Edge`, :code:`Vertex`.
   * - :code:`Location[T]`
     - A location, i.e. a specific cell, edge or vertex, where :code:`T` is a :code:`LocationType`. Locations can only be constructed in :code:`LocationSpecifications` or :code:`LocationComprehensions`.
   * - :code:`DataType`
     - A scalar value
        - :code:`bool`
        - :code:`int`
        - :code:`float`

Variable types:

- :code:`Field`
- :code:`TemporaryField`
- :code:`Location`
- :code:`Mesh`

Todo: Expand & seperate variable and type documentation.

Statements
==========

The only statements allowed are assignments.

Assignments
-----------

The left-hand-side of an assignment is always a :code:`Field` defined on the current iteration space. If the field is not passed as a stencil argument, a temporary field is automatically introduced and may be referenced throughout the entire stencil. The right-hand-side of an assignment is an expression with type :code:`DataType`.

.. code-block:: python

  field = expression
  field[location] = expression

Modified example of the copy stencil emphasizing the behaviour of temporary fields:

.. code-block:: python

  @gtscript.stencil
  def tmp_field_copy(
    mesh: Mesh,
    field_in : Field[[Vertex], float],
    field_out : Field[[Vertex], float]
  ):
      with location(Vertex) as v:
        tmp_field[v] = field_in[v]
      with location(Vertex) as v:
        field_out[v] = tmp_field[v]

Todo: Off-center writes

Expressions
===========

Literals / Constants
--------------------

Only boolean and numeric literals are allowed. The precision of numeric literals is contrary to python by default machine-independent, but may be overridden by the user by specifying the :code:`dtype` stencil decorator argument.

.. code-block:: python

  # booleans are always of type bool
  True
  False
  # integer are of type dtypes["int"]
  3
  # float are of type dtypes["float"]
  3.

The user may for example use 32 bit float and integer values for all literals of a stencil as follows:

.. code-block:: python

  @gtscript.stencil(dtypes={"float": np.float32, "int": np.int32})
  def my_stencil(mesh: gtscript.Mesh):
    with location(Vertex) as v:
        my_field = my_field + 1.1 # 32 bit approximation of 1.1 used
        my_field = my_field + 1   # 32 bit integer with value 1 used

The user may further explicitly specify the type of a literal using regular instantiation syntax, e.g. :code:`float32("1.1")` for a float with 32 bits of precision.

.. code-block:: python

  # Integer
  uint32("1")
  uint64("1")
  int32("1")
  int64("1")
  # Floating point
  float32("1.1")
  float64("1.1")

This allows for usage of literals with mixed precision.

.. code-block:: python

  @gtscript.stencil(dtypes={"float": np.float32})
  def my_stencil(mesh: gtscript.Mesh):
    with location(Vertex) as v:
        my_field = my_field + 1.1 + float64("1.1")

Field access
------------

Fields are accessed using the subscript operator :code:`[]` with the index being the location to be accessed and a vertical offset. If no subscript is provided the value at the current location and layer is retrieved.

.. code-block:: python

  field        # value at the current primary location and layer
  field[v, 0]  # value at the current layer and location `v`
  field[v, -1] # value at location `v` with vertical offset -1

Arithmetic operators
--------------------

Arithmetic operators on values of type :code:`gtc.common.DataType` follow the regular python syntax.

.. code-block:: python

  a + b
  a - b
  a * b
  a / b

Neighbor reductions
--------------------

Reductions over neighbors are composed of a reduction function, a generator expression, representing a set of values on the neighboring locations, and a neighbor selector, specifying the neighbors to be reduced over. GTScript supports four reduction functions :code:`sum`, :code:`product`, :code:`min`, :code:`max`, computing the sum, product, mimimum and maximum, respectively, of its arguments. The argument to a reduction function is a generator expression with the following syntax:

.. code-block:: python

  expression for location in neighbor_selector


where :code:`expression` is just an expression, :code:`location` the name of the symbol referencing the neighbors location and :code:`neighbor_selector` is a neighbor selector. Inside the expression, fields may be referenced using :code:`location`. Neighbors of the primary location can be selected via calls to the built-in function :code:`neighbors` or one of the convenience functions `vertices` and `edges`.

.. code-block:: python

  # signature
  neighbors(primary_location : Location, *chain : LocationType)

  # select all cells sharing a common vertex with the current `cell`
  neighbors(cell, Vertex, Cell)

Pseudo-code for :code:`vertices` and :code:`edges` convenience functions:

.. code-block:: python

  def vertices(of : Location):
    return neighbors(of, Vertex)

  def edges(of : Location):
    return neighbors(of, Edge)

Example computing the sum of :code:`vertex_field` over all neighboring vertices of :code:`e`:

.. code-block:: python

  sum(vertex_field[v] for v in vertices(e))
  product(vertex_field[v] for v in vertices(e))

Todo: Sparse field example
