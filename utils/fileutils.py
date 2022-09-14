import os

_default_path = "~/Main/midjourney/"
_tmp_path = "~/Main/tmp/"


def set_default_path(new_path):
    global _default_path
    _default_path = new_path


def set_tmp_path(new_path):
    global _tmp_path
    _tmp_path = new_path


def get_default_path(*args, **kwargs):
    return get_path_in_dir(os.path.expanduser(_default_path), *args, **kwargs)


def get_tmp_path(*args, **kwargs):
    return get_path_in_dir(os.path.expanduser(_tmp_path, *args, **kwargs))


def get_path_in_dir(base_dir, subdir=None, file_name=None, create_dirs=True):
    path = base_dir
    if subdir:
        if subdir.startswith(base_dir):
            subdir.replace(base_dir, "")

        path = os.path.join(path, subdir)
    if create_dirs:
        os.makedirs(path, exist_ok=True)

    if file_name:
        path = os.path.join(path, file_name)

    return path
