import contextlib
import torch
import functools

# Cache the default device to avoid repeated checks
_cached_device = None

def get_default_device():
    global _cached_device
    if _cached_device is None:
        if torch.cuda.is_available():
            _cached_device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            _cached_device = torch.device("mps")
        else:
            _cached_device = torch.device("cpu")
    return _cached_device

def safe_autocast_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = get_default_device()
            if device.type in ["cuda", "cpu"]:
                with torch.amp.autocast(device_type=device.type, enabled=enabled):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextlib.contextmanager
def safe_autocast(enabled=True):
    device = get_default_device()
    if device.type in ["cuda", "cpu"]:
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            yield
    else:
        yield  # MPS or other unsupported backends skip autocast
