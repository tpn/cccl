# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the Mamba v1 selective scan with one block per (batch, channel) row. Each time step is an 
   affine map packed into a uint64; a block-wide inclusive scan composes them, and a carry links 
successive tiles.
"""
from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.cuda import libdevice

import cuda.coop.numba_mlir as coop

THREADS = 128
ITEMS_PER_THREAD = 4
MAX_STATE = 256


@cuda.jit(device=True)
def pack_pair(a, b):
    lo = types.uint64(types.uint32(libdevice.float_as_int(a)))
    hi = types.uint64(types.uint32(libdevice.float_as_int(b)))
    return lo | (hi << types.uint64(32))


@cuda.jit(device=True)
def pair_a(pair):
    return libdevice.int_as_float(types.int32(types.uint32(pair)))


@cuda.jit(device=True)
def pair_b(pair):
    return libdevice.int_as_float(types.int32(types.uint32(pair >> types.uint64(32))))


@cuda.jit(device=True)
def apply_softplus(s):
    """Softplus with the threshold at 20: log(1 + exp(s)) for s <= 20, otherwise s"""
    if s > types.float32(20.0):
        return s
    return libdevice.log1pf(libdevice.expf(s))

@cuda.jit(device=True)
def seq_index(row, length, timestep):
    return row * length + timestep

@cuda.jit(device=True)
def a_index(channel, nstate, n):
    return channel * nstate + n

@cuda.jit(device=True)
def state_index(row, nstate, n):
    return row * nstate + n

@cuda.jit(device=True)
def weight_index(batch, channel, n, timestep, dim, nstate, length, groups, variable):
    if not variable:
        return channel * nstate + n        
    g = channel // (dim // groups)
    return ((batch*groups + g)*nstate + n)*length + timestep

@cuda.jit(device=True)
def silu(x):
    return x / (types.float32(1.0) + libdevice.expf(-x))


@cuda.jit(device=True)
def compose(left, right):
    """Compose two packed affine maps, applying left first.
    
    @Returns:
        uint64: The coefficients of the combined affine transformation.
    """
    a0 = libdevice.int_as_float(np.int32(np.uint32(left)))
    b0 = libdevice.int_as_float(np.int32(np.uint32(left >> np.uint64(32))))
    a1 = libdevice.int_as_float(np.int32(np.uint32(right)))
    b1 = libdevice.int_as_float(np.int32(np.uint32(right >> np.uint64(32))))
    alpha = a1*a0
    beta = a1*b0+b1
    alpha_int64 = types.uint64(types.uint32(libdevice.float_as_int(alpha)))
    beta_int64 = types.uint64(types.uint32(libdevice.float_as_int(beta)))
    return alpha_int64 | (beta_int64 << types.uint64(32))


@cuda.jit
def selective_scan_kernel(
    u,
    delta,
    a,
    b,
    c,
    d,
    z,
    bias,
    output,
    last_state,
    dim,
    length,
    nstate,
    b_groups,
    c_groups,
    variable_b,
    variable_c,
    has_d,
    has_z,
    has_bias,
    softplus,
    items_per_thread,
):
    block = coop.this_block()
    thread = cuda.threadIdx.x
    row = cuda.blockIdx.x
    batch, channel = row // dim, row % dim
    u_td = coop.ThreadData(items_per_thread)
    delta_td = coop.ThreadData(items_per_thread)
    pairs = coop.ThreadData(items_per_thread)
    agg = coop.ThreadData(items_per_thread=1)
    y_td = coop.ThreadData(items_per_thread)

    n = 0
    A_value = a[a_index(channel, nstate, n)]
    B_value = b[weight_index(batch, channel, n, 0, dim, nstate, length, b_groups, variable_b)]
    C_value = c[weight_index(batch, channel, n, 0, dim, nstate, length, c_groups, variable_c)]

    h_in = types.float32(0.0)
    tile_items = THREADS * items_per_thread
    ntiles = (length + tile_items - 1) // tile_items

    for tile in range(ntiles):
        start = tile * tile_items
        valid = min(length - start, tile_items)

        coop.load(block, u, u_td, offset=seq_index(row, length, start), algorithm="warp_transpose", valid_items=valid, oob_default=0)
        coop.load(block, delta, delta_td, offset=seq_index(row, length, start), algorithm="warp_transpose", valid_items=valid, oob_default=0)

        for i in range(items_per_thread):
            alpha = libdevice.expf(delta_td[i] * A_value)
            beta = delta_td[i] * B_value * u_td[i]
            t = start + thread*items_per_thread + i
            if t < length:
                pairs[i] = types.uint64(pack_pair(alpha, beta))
            else:
                pairs[i] = types.uint64(pack_pair(types.float32(1.0), types.float32(0.0)))

        scan_result = coop.inclusive_scan(
            block,
            pairs,
            scan_op=compose,
            algorithm="warp_scans",
            aggregate_output=agg
        )
        for i in range(items_per_thread):
            p = scan_result[i]
            h = pair_a(p) * h_in + pair_b(p)
            y_td[i] = types.float32(C_value * h)

        coop.store(block, output, y_td, offset=seq_index(row, length, start), algorithm="warp_transpose", valid_items=valid)
        h_in = pair_a(agg[0]) * h_in + pair_b(agg[0])
        block.sync()

    if thread == 0:
        last_state[state_index(row, nstate, n)] = h_in


def selective_scan(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, items_per_thread= ITEMS_PER_THREAD
):
    """Return output and canonical final state for finite float32 inputs.

    u/delta: (batch, dim, length); A: (dim, state), 1 <= state <= 256.
    B/C: (dim, state), (batch, state, length), or (batch, groups, state, length).
    Groups must divide dim. Optional D/bias: (dim,); z: u.shape.
    This teaching wrapper copies arrays and synchronizes each invocation.
    """

    def checked(name, value, shape=None):
        value = np.asarray(value)
        if value.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32")
        if shape is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        return np.ascontiguousarray(value)

    u = checked("u", u)
    if u.ndim != 3 or any(s < 1 for s in u.shape):
        raise ValueError("u must have positive (batch, dim, length) dimensions")
    batch, dim, length = u.shape
    delta = checked("delta", delta, u.shape)
    A = checked("A", A)
    if A.ndim != 2 or A.shape[0] != dim or not 1 <= A.shape[1] <= MAX_STATE:
        raise ValueError("A must have shape (dim, state), with 1 <= state <= 256")
    nstate = A.shape[1]

    def weights(name, value):
        value = checked(name, value)
        if value.shape == (dim, nstate):
            return value, False, 1
        if value.shape == (batch, nstate, length):
            return value, True, 1
        if (
            value.ndim == 4
            and value.shape[0] == batch
            and value.shape[2:] == (nstate, length)
            and value.shape[1] > 0
            and dim % value.shape[1] == 0
        ):
            return value, True, value.shape[1]
        raise ValueError(f"unsupported {name} shape {value.shape}")

    B, variable_b, b_groups = weights("B", B)
    C, variable_c, c_groups = weights("C", C)
    optional = []
    for name, value, shape in (
        ("D", D, (dim,)),
        ("z", z, u.shape),
        ("delta_bias", delta_bias, (dim,)),
    ):
        optional.append(
            np.zeros(1, dtype=np.float32)
            if value is None
            else checked(name, value, shape)
        )
    host_inputs = [u, delta, A, B, C, *optional]
    device_inputs = [cuda.to_device(value.reshape(-1)) for value in host_inputs]
    output = cuda.device_array(u.size, dtype=np.float32)
    last = cuda.device_array(batch * dim * nstate, dtype=np.float32)
    selective_scan_kernel[batch * dim, THREADS](
        *device_inputs,
        output,
        last,
        dim,
        length,
        nstate,
        b_groups,
        c_groups,
        variable_b,
        variable_c,
        D is not None,
        z is not None,
        delta_bias is not None,
        bool(delta_softplus),
        items_per_thread,
    )
    return output.copy_to_host().reshape(u.shape), last.copy_to_host().reshape(
        batch, dim, nstate
    )


def _expand_weight(weight, batch, dim, nstate, length):
    """Broadcast B or C to (batch, dim, nstate, length)."""
    if weight.shape == (dim, nstate):
        return np.broadcast_to(weight[None, :, :, None], (batch, dim, nstate, length))
    if weight.ndim == 3:
        return np.broadcast_to(weight[:, None], (batch, dim, nstate, length))
    channels_per_group = dim // weight.shape[1]
    return weight[:, np.arange(dim) // channels_per_group]


def selective_scan_reference(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False
):
    """Serial float64 reference with the same arguments as ``selective_scan``."""
    u = u.astype(np.float64)
    batch, dim, length = u.shape
    A = A.astype(np.float64)
    nstate = A.shape[1]
    s = delta.astype(np.float64)
    if delta_bias is not None:
        s = s + delta_bias.astype(np.float64)[None, :, None]
    if delta_softplus:
        s = np.where(s <= 20.0, np.log1p(np.exp(np.minimum(s, 20.0))), s)
    B = _expand_weight(B.astype(np.float64), batch, dim, nstate, length)
    C = _expand_weight(C.astype(np.float64), batch, dim, nstate, length)

    alpha = np.exp(s[:, :, None, :] * A[None, :, :, None])
    beta = (s * u)[:, :, None, :] * B
    h = np.zeros((batch, dim, nstate))
    v = np.empty_like(u)
    for t in range(length):
        h = alpha[..., t] * h + beta[..., t]
        v[..., t] = np.sum(C[..., t] * h, axis=-1)
    if D is not None:
        v = v + D.astype(np.float64)[None, :, None] * u
    if z is not None:
        z = z.astype(np.float64)
        v = v * z / (1.0 + np.exp(-z))
    return v, h


def run_example(items_per_thread: int = ITEMS_PER_THREAD) -> float:
    """Check several sequence lengths against the reference; return the max error."""

    rng = np.random.default_rng(0)
    batch, dim, nstate = 2, 3, 1
    tile_items = THREADS * items_per_thread
    worst = 0.0
    for length in (31, tile_items, tile_items + 1, 2 * tile_items + 1):
        u = rng.standard_normal((batch, dim, length), dtype=np.float32)
        delta = rng.uniform(0.01, 0.1, (batch, dim, length)).astype(np.float32)
        A = -rng.uniform(0.5, 2.0, (dim, nstate)).astype(np.float32)
        B = rng.standard_normal((dim, nstate), dtype=np.float32)
        C = rng.standard_normal((dim, nstate), dtype=np.float32)

        output, last_state = selective_scan(
            u, delta, A, B, C, items_per_thread=items_per_thread
        )
        want_output, want_state = selective_scan_reference(u, delta, A, B, C)
        np.testing.assert_allclose(output, want_output, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(last_state, want_state, rtol=1e-4, atol=1e-5)
        worst = max(worst, float(np.max(np.abs(output - want_output))))
    return worst


def main() -> int:
    for items_per_thread in (1, 4):
        error = run_example(items_per_thread)
        print(f"items_per_thread={items_per_thread}: max output error {error:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
