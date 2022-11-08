import random
import re

import numpy as np
from utils import ReTerm


class UserPrompt:
    def __init__(self, prompt: "StableDiscoPrompt", img_url: str, date_str: str, user: str) -> None:
        self.prompt = prompt
        self.img_url = img_url
        self.date_str = date_str
        self.user = user

    def get_prompt(self) -> str:
        return self.prompt.prompt

    def get_args(self) -> str:
        return self.prompt.args

    def get_prompt_with_args(self):
        arg_str = " ".join([f"{key} {val}" for key, val in self.prompt.args.items()])
        return f"{self.prompt.prompt} {arg_str}"

    @staticmethod
    def from_discord_msg(discord_json):
        content = re.sub(r"\s", " ", discord_json["content"])
        prompt_str = re.findall(r'(?<=!dream ").*(?=` )', content)
        if not prompt_str:
            prompt_str = re.findall(r'(?<=!dream ").*(?=" -)', content)
        try:
            prompt = StableDiscoPrompt.from_str(prompt_str[0])
        except Exception as ex:
            print(f"Failed to extract discord prompt: {ex}")
            print(content)

        if not discord_json["attachments"]:
            print(discord_json.keys())
            print(discord_json["attachments"])
            print(discord_json["embeds"])
            raise Exception("No attachment")

        user = discord_json["mentions"][0]["name"]

        return UserPrompt(
            prompt,
            discord_json["attachments"][0]["url"],
            discord_json["timestamp"],
            user,
        )

    def __str__(self):
        print(type(self))
        return f"Date:{self.date_str}\n{self.prompt}\nImage Url: {self.img_url}"


class StableDiscoPrompt:
    def __init__(self, prompt, args):
        self.prompt = prompt
        self.args = args

    def get_prompt_with_args(self):
        return f"{self.prompt} {self.get_arg_str()}"

    def get_arg_str(self):
        return " ".join([f"{key} {val}" for key, val in self.args.items()])

    @classmethod
    def from_str(cls, full_prompt):
        prompt, args = arg_prompt_split(full_prompt)
        return cls(prompt, args)

    def __str__(self):
        return f"prompt: {self.prompt} args:{self.get_arg_str()}"


class ArgSelector:
    def __init__(self, arg_val_cnt, total_prompt_cnt):
        self._arg_val_freq = {key: val for key, val in arg_val_cnt.items() if key not in {"--h", "--w"}}
        self._param_freq = {param: sum(val.values()) / total_prompt_cnt for param, val in self._arg_val_freq.items()}
        for param, vals_cnts in self._arg_val_freq.items():
            total_cnt = sum(vals_cnts.values())
            for param_val in vals_cnts:
                vals_cnts[param_val] /= total_cnt

    # prob_boost increases the chance of arguments to account for not always supporting them
    def get_random_args(self, prob_boost=10000.0):
        args = []
        for param, prob in self._param_freq.items():
            if random.random() < prob * prob_boost:
                value = self._choose_arg_value(param)
                args.append(f"{param} {value}")

        return " ".join(args)

    def _choose_arg_value(self, param):
        freqs = self._arg_val_freq[param]
        keys = list(freqs.keys())
        probs = [freqs[key] for key in keys]
        return np.random.choice(keys, p=probs)


def arg_prompt_split(command):
    args_iters = ReTerm.sd_args_regex.finditer(command)

    args = [command[it.start() : it.end()].strip() for it in args_iters]
    prompt = re.sub("|".join(args), "", command).strip()

    command_mapper = {
        "-width": "-W",
        "-height": "-H",
        "-cfg": "-C",
        "-cfg_scale": "-C",
        "-seed": "-S",
        "-steps": "-s",
        "-sampler": "-A",
        "-prior": "-p",
        "-ascii": "-a",
        "-separate-images": "-i",
        "-grid": "-g",
        "-number": "-n",
        "-tokenize": "-t",
    }
    no_value = {"-a", "-t", "-i", "-g", "-t", "-p"}

    recognized_args = command_mapper.keys() | command_mapper.values()

    def split_key_val(arg):
        split = [part.strip() for part in re.split("[ =]", arg.strip())]
        split[0] = re.sub(" +", " ", split[0])
        split[0] = re.sub("–", "-", split[0])
        split[0] = re.sub("--", "-", split[0]).strip()

        without_nums = re.sub(r"\d+", "", split[0])
        if without_nums in recognized_args and without_nums != split[0]:
            nums = re.findall(r"\d+", split[0])
            split.append(nums[0])
            split[0] = without_nums

        if split[0].lower() in command_mapper:
            split[0] = command_mapper[split[0].lower()]

        if len(split) != 2:
            return split[0], ""

        return split[0], split[1]

    key_val_pairs = [split_key_val(arg) for arg in args]
    args_to_remove = set()
    for idx, arg_pair in enumerate(key_val_pairs):
        key, val = arg_pair

        if key not in recognized_args or (not val and key not in no_value):
            # Add the origional arg to the prompt. This is a common typo
            # which affects the tokens
            prompt += f" {args[idx]}"
            args_to_remove.add(idx)

    key_val_pairs = [pair for idx, pair in enumerate(key_val_pairs) if idx not in args_to_remove]

    args = {entry[0]: entry[1] for entry in key_val_pairs if len(entry) == 2}
    return prompt, args
