"""pkit — the personality project's analysis surface.

Light by default: `import pkit` pulls numpy/pandas but NOT torch.
The extraction layer (model loading, logprob readouts) lives in
pkit.extraction and is imported lazily on first attribute access.

    import pkit
    R = pkit.load.self_matrix()            # models x 525 framing-mean EVs
    J = pkit.load.load_judge("Qwen7")      # B[a,b] = EV of P(b|a)
    H = pkit.load.human_corr()             # human 525 x 525
    fit = pkit.cooking.cook("Llama")       # base-rate LS fit -> phi
"""
from . import axes, cooking, load, measures, paths, roster  # noqa: F401

__all__ = ["axes", "cooking", "extraction", "load", "measures",
           "paths", "roster"]


def __getattr__(name):
    if name == "extraction":
        from . import extraction
        return extraction
    raise AttributeError(f"module 'pkit' has no attribute {name!r}")
