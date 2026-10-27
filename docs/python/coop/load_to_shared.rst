.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-load-to-shared:

Global-to-shared copies
=========================

Use :func:`cuda.coop.load_to_shared_async` to stage a contiguous global-memory
range for threads in a block to reuse. It issues and commits a transfer into
caller-owned shared memory. Its pending result separates issue from completion:

.. code-block:: python

   pending = coop.load_to_shared_async(
       block, source, shared_buffer, num_items=tile_items, offset=tile_offset
   )
   # Compute values that do not depend on the destination buffer.
   shared_values = pending.wait()
   # Read the copied values through shared_values.

``num_items`` counts the whole block's transfer and is known at compilation.
``offset`` selects the source range and can vary by block at runtime. All
threads in the block must agree on the arguments. Block sizes are multiples
of 32, and every thread executes both issue and wait.

The operation is useful when several threads consume the same staged values:
for example, a tile's neighborhood calculation can read adjacent shared
elements after the copy completes. Independent arithmetic between issue and
wait can overlap the transfer on supported hardware. The API makes no
performance guarantee for a particular tile or device.

Choosing a buffer
-----------------

The destination is a contiguous one-dimensional shared array or tensor with
the source dtype. Its base must be aligned to at least 16 bytes. A typed
``TempStorage.reserve(..., alignment=16)`` allocation provides this alignment;
keep its descriptor separate from scratch used by other operations while the
copy or its consumers are live.

Simultaneously pending copies need disjoint destination ranges. Use separate
descriptors or ``TempStorage(sharing="exclusive")`` for their reservations;
the default ``sharing="shared"`` permits reservations to reuse the same bytes.

For a native Numba allocation, request the alignment explicitly:
``cuda.shared.array(buffer_items, dtype=types.int32, alignment=16)``. Its
default alignment is insufficient for this operation. For a native CuTe
allocation, use ``SmemAllocator().allocate_tensor`` with
``byte_alignment=16``. These are allocation options; the transfer itself
does not strengthen the alignment of an existing buffer.

For ``N > 0`` elements of size ``S`` bytes, reserve at least
``round_up(N * S, 16) + 16`` bytes. A zero-element copy requires no data bytes,
but still participates in issue and wait. For example, a block staging 128
``float32`` elements needs a buffer of 132 elements (528 bytes).

This padding allows CUB to handle a source whose address is only aligned to
its element type. The actual copied range can start after head padding in
the buffer. ``pending.wait()`` returns a view beginning at that exact location,
with the source dtype and ``num_items`` elements. Use this returned view to
read the data. Its address may have weaker alignment than the buffer base.
The remainder of the allocation is padding, not additional copied elements.

The caller retains ownership of the destination allocation. Keep the source
unchanged and the destination untouched until wait completes. After consumers
finish reading the returned view, synchronize the block before overwriting
or reusing the buffer. Waiting for the producer does not synchronize future
consumer reads.

Lifetime and control flow
-------------------------

Each pending copy must be waited on exactly once. Assigning an alias to the
handle does not create another transaction. The first implementation supports
issue and wait in the kernel's outer region; calls inside runtime branches
or loops are rejected. Independent runtime computation may appear between
them, provided it does not use the pending handle or destination.

The compiler allocates dedicated barrier storage for the transaction. Ordinary
cooperative primitives between issue and wait cannot reuse that storage. Wait
completes the copy and retires its barrier state. It returns a native shared
view, so subsequent code can use the selected DSL's ordinary indexing and
shared-memory operations.

Relationship to Load and Store
------------------------------

``coop.load`` fills a per-thread payload, and ``coop.store`` writes a
per-thread payload to a memory destination.
``load_to_shared_async`` copies a whole contiguous range directly into a
shared buffer and exposes completion through its pending handle. It has no
``items_per_thread`` argument or blocked/striped output layout.

An ordinary Load's output parameter still denotes a supported per-thread
payload. This operation does not make a native shared array a valid output
for ``coop.load``. Use the explicit global-to-shared operation for staging,
then use the returned native shared view after waiting.

Examples
--------

Both examples stage a different source range for each block, perform
independent work, then read neighboring threads' values from the returned
shared view. The examples are exercised with ``items_per_thread`` equal to
1 and 4; this variable controls the example's work distribution, while
``num_items`` remains the whole block's transfer size.

Numba-CUDA-MLIR
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_load_to_shared.py
   :language: python
   :start-after: # load-to-shared-example-begin
   :end-before: # load-to-shared-example-end
   :dedent: 4

CUTLASS DSL
^^^^^^^^^^^

The example uses ``cutlass``, ``cutlass.cute``, NumPy, and the test suite's
``device_array`` host allocation helper.

.. literalinclude:: ../../../python/cuda_coop/tests/backends/cutlass/runtime/test_load_to_shared.py
   :language: python
   :start-after: # load-to-shared-cutlass-example-begin
   :end-before: # load-to-shared-cutlass-example-end
   :dedent: 4

Hardware paths
--------------

The implementation binds CUB's ``detail::BlockLoadToShared``. On Hopper and
newer GPUs, CUB uses linear ``cp.async.bulk`` transfers and an ``mbarrier``.
This path does not require a multidimensional tensor-map descriptor. Ampere
uses ``cp.async`` with its completion waits, and older GPUs use a synchronous
fallback. Every path honors the same wait contract.

See the :doc:`CCCL TMA guide <../../cccl/tma>` for the underlying C++ facilities.
