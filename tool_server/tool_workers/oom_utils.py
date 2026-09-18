"""Shared CUDA OOM classification for tool workers and their clients."""

CUDA_OOM_ERROR_CODE = 50002
CUDA_OOM_ERROR_TYPE = "cuda_oom"

# Some CUDA libraries report allocation failures as RuntimeError instead of
# torch.cuda.OutOfMemoryError. Keep this list shared so the worker response and
# the agent-side retry decision cannot silently disagree.
CUDA_OOM_MESSAGE_PATTERNS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda malloc failed",
)


def message_indicates_cuda_oom(message):
    normalized = str(message).lower()
    return any(pattern in normalized for pattern in CUDA_OOM_MESSAGE_PATTERNS)


def response_indicates_cuda_oom(response):
    """Return whether a serialized worker error represents a CUDA OOM."""
    if not isinstance(response, dict):
        return False

    error_code = response.get("error_code", -1)
    try:
        error_code = int(error_code)
    except (TypeError, ValueError):
        error_code = -1

    error_type = str(response.get("error_type", "")).lower()
    return (
        error_code == CUDA_OOM_ERROR_CODE
        or error_type == CUDA_OOM_ERROR_TYPE
        or message_indicates_cuda_oom(response.get("message", ""))
    )
