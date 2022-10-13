import os


def add_env_var():
    default_values = {"NUMEXPR_MAX_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "DISPLAY": "0"}
    for key, value in default_values.items():
        os.environ[key] = os.environ.get(key, value)

