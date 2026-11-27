# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-wise QSA prefill scoring for Hygon.

Control variant for the DCU backend: the generic implementation scores
BLOCK_R rows of a request with a tiled ``tl.dot``.  This module keeps the
generic host pipeline (input validation, chunked logits workspace and top-k
selection) and only replaces the score launcher with a per-row, per-head
elementwise kernel that mirrors the vendor AMD implementation.
"""

import importlib

import triton
import triton.language as tl

# ``flaggems_vllm.ops`` re-exports the function under the same name as the
# module, so ``import ... as`` would bind the function; import by name to get
# the module object whose ``_prefill_logits`` we replace.
_generic = importlib.import_module("flaggems_vllm.ops.qsa_select_paged_prefill")

_BLOCK_N = 32
_NUM_WARPS = 4


@triton.jit
def _prefill_rows_score_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    query_offset,
    logits_width,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    first = tl.load(query_start_loc_ptr)
    token = first + query_offset + row
    request = tl.zeros((), tl.int32)
    for r in tl.static_range(0, NUM_REQUESTS):
        request += (token >= tl.load(query_start_loc_ptr + r + 1)).to(tl.int32)
    visible = tl.load(visible_ptr + query_offset + row)
    if tile * BLOCK_N >= visible:
        tl.store(
            logits_ptr + row * stride_logits_row + cols,
            tl.full((BLOCK_N,), -float("inf"), tl.float32),
            mask=cols < logits_width,
        )
        return
    live = (cols < logits_width) & (cols < visible)
    logical_page = cols // PAGE_SIZE
    page_offset = cols % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr + request * stride_table_req + logical_page * stride_table_page,
        mask=live & (logical_page < PAGE_TABLE_WIDTH),
        other=-1,
    )
    live &= (physical_page >= 0) & (physical_page < num_pages)
    keys = tl.load(
        k_cache_ptr
        + physical_page[:, None].to(tl.int64) * stride_cache_block
        + page_offset[:, None] * stride_cache_token
        + dims[None, :],
        mask=live[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    score = tl.zeros((BLOCK_N,), tl.float32)
    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + (query_offset + row) * stride_q_row + head * stride_q_head + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    tl.store(
        logits_ptr + row * stride_logits_row + cols,
        tl.where(live, score * SCALE, -float("inf")),
        mask=cols < logits_width,
    )


def _prefill_logits_rows(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    logits_workspace,
    max_query_len,
    logits_width,
    query_offset,
    num_queries,
    *,
    launch_metadata=None,
):
    """Score launcher with the same contract as the generic ``_prefill_logits``."""
    if logits_workspace.ndim == 2:
        logits = (
            logits_workspace
            if num_queries == logits_workspace.shape[0]
            else logits_workspace[:num_queries]
        )
    else:
        logits = logits_workspace[: num_queries * logits_width].view(
            num_queries, logits_width
        )
    if num_queries == 0:
        return logits
    grid = (num_queries, triton.cdiv(logits_width, _BLOCK_N))
    _prefill_rows_score_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        query_offset,
        logits_width,
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        NUM_REQUESTS=query_start_loc.numel() - 1,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
        SCALE=q.shape[2] ** -0.5,
        num_warps=_NUM_WARPS,
    )
    return logits


# Swap the score launcher inside the generic host pipeline; the vendor
# dispatch in ``flaggems_vllm/__init__.py`` then re-exports the resulting
# function as the top-level ``flaggems_vllm.qsa_select_paged_prefill``.
_generic._prefill_logits = _prefill_logits_rows

qsa_select_paged_prefill = _generic.qsa_select_paged_prefill

__all__ = ["qsa_select_paged_prefill"]
