import re


class ReTerm:
    num_regex = re.compile(r"[-+]?(([0-9]+\.?[0-9]*)|(\.[0-9]+))")
    weight_regex = re.compile(r"::((([0-9]+\.?[0-9]*)|(\.[0-9]+))?|[-+](([0-9]+\.?[0-9]*)|(\.[0-9]+)))")
    arg_regex = re.compile(r"(?<!\S)-[a-z]+([ a-z0-9\.=;]|:[^:])+(?!\S-)")
    sd_args_regex = re.compile(r" ([-–]|[-–]{2})\w+ *=?[\w\.]*(?= *([ -–]|$))")
    # (?<!\S)-[a-z]+([ a-z0-9]|:[^:])+(?!\S-)


def round_to_multiple(x, base):
    return base * round(x / base)


def print_list_lines(lst, num_lines=2):
    lines = "\n" * num_lines
    print(lines.join(lst))


def remove_none_and_dup(lst):
    lst = list(filter(None, lst))
    return remove_dup(lst)


def remove_dup(lst):
    return list(set(lst))


def flatten_nested_dict(nested):
    ret = {}

    for key, inner_dict in nested.items():
        ret[key] = dict(inner_dict)
    return {key: dict(inner_dict) for key, inner_dict in nested.items()}


def sort_num_asc(lst, inplace=True):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    if inplace:
        lst.sort(key=alphanum_key)
        return lst
    return sorted(lst, key=alphanum_key)
