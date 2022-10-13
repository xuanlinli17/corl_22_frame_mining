from .converter import as_dtype


def f64_to_f32(item):
    """
    Convert all float64 data to float32
    """
    from .type_utils import get_dtype
    from .converter import as_dtype

    sign = get_dtype(item) in ["float64", "double"]
    return as_dtype(item, "float32") if sign else item


def to_f32(item):
    return as_dtype(item, "float32")


def to_f16(item):
    return as_dtype(item, "float16")

