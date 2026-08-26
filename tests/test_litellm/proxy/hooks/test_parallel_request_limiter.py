"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

from datetime import datetime

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.route_llm_request import ProxyModelNotFoundError
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import EmbeddingResponse, TextCompletionResponse, Usage


@pytest.mark.parametrize(
    "response_obj",
    [
        EmbeddingResponse(
            model="text-embedding-3-small",
            usage=Usage(prompt_tokens=50, completion_tokens=0, total_tokens=50),
        ),
        TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_log_success_event_counts_non_chat_response_tokens(response_obj):
    """
    Embedding and text completion responses must increment the per key, user,
    team, and end user TPM counters, not just chat completion ModelResponse
    objects.
    """
    _api_key = hash_token("sk-12345")
    user_id = "ishaan"
    team_id = "litellm-team"
    end_user_id = "customer-1"

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"

    scope_ids = [_api_key, user_id, team_id, end_user_id]
    for scope_id in scope_ids:
        await parallel_request_handler.internal_usage_cache.async_set_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
            litellm_parent_otel_span=None,
        )

    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "user_api_key_user_id": user_id,
                "user_api_key_team_id": team_id,
                "user_api_key_model_max_budget": {},
            }
        },
        "user": end_user_id,
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    for scope_id in scope_ids:
        current = await parallel_request_handler.internal_usage_cache.async_get_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            litellm_parent_otel_span=None,
        )
        assert current["current_tpm"] == 50, (
            f"expected 50 tokens counted for {scope_id}, "
            f"got {current['current_tpm']}"
        )


@pytest.mark.asyncio
async def test_model_not_found_releases_reserved_slot_instead_of_leaking():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/18060.

    A request that passes async_pre_call_hook (reserving a concurrency slot) but
    is then rejected during routing because the model name doesn't exist
    (ProxyModelNotFoundError) never reaches an actual LLM call, so
    async_log_failure_event never fires to release the slot. Before the fix,
    that leaked slot stuck around until the per-minute cache key naturally
    expired, and a steady trickle of typo'd-model requests kept refreshing that
    TTL, permanently pinning the key at its max_parallel_requests limit and
    429-ing every subsequent request -- including valid ones.
    """
    api_key = hash_token("sk-model-not-found")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=1)

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    # First request: passes the pre-call check, reserving the key's one slot.
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data={"model": "Model1"},
        call_type="completion",
    )

    # It then fails at routing because "Model1" doesn't exist.
    await parallel_request_handler.async_post_call_failure_hook(
        request_data={"model": "Model1"},
        original_exception=ProxyModelNotFoundError(route="/chat/completions", model_name="Model1"),
        user_api_key_dict=user_api_key_dict,
    )

    # A second, unrelated request must not be wrongly rejected: the failed
    # first request's slot must have been released, not leaked.
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data={"model": "model-abcd"},
        call_type="completion",
    )


@pytest.mark.asyncio
async def test_unrelated_failure_does_not_release_slot_twice():
    """
    async_post_call_failure_hook must stay scoped to ProxyModelNotFoundError.

    Any other failure (e.g. an actual LLM API error) already gets its slot
    released by async_log_failure_event via litellm's own logging pipeline; if
    async_post_call_failure_hook also released it here, the slot would be
    double-released, letting one extra concurrent request slip past
    max_parallel_requests.
    """
    api_key = hash_token("sk-unrelated-failure")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=1)

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data={"model": "gpt-3.5-turbo"},
        call_type="completion",
    )

    await parallel_request_handler.async_post_call_failure_hook(
        request_data={"model": "gpt-3.5-turbo"},
        original_exception=HTTPException(status_code=500, detail="upstream provider error"),
        user_api_key_dict=user_api_key_dict,
    )

    # The slot from the first request is still reserved (as it should be until
    # async_log_failure_event releases it), so a second concurrent request
    # must be rejected.
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-3.5-turbo"},
            call_type="completion",
        )
    assert exc_info.value.status_code == 429
