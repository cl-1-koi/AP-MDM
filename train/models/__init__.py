from . import dit
from . import ema

# Sudoku uses the DIT backbone.  The released package imported the unrelated
# autoregressive backend eagerly, which made DIT unusable when FlashAttention
# was absent even though DIT now has the SDPA compatibility path.
try:
    from . import autoregressive
except (ImportError, OSError) as exc:
    if "flash_attn" not in str(exc):
        raise
    autoregressive = None
