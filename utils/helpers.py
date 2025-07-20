import os


def is_runpod():
    return os.getenv("RUNPOD_POD_ID") is not None or os.path.exists("/workspace")

def is_kaggle():
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

def is_local():
    return not is_runpod() and not is_kaggle()

def get_base_path():
    if is_runpod():
        return "/workspace"
    elif is_kaggle():
        return "/kaggle/input"
    else:
        return "./data"