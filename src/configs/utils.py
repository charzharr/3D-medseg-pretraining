
import os
import warnings
import pathlib
import yaml
import pprint
import numpy as np



def get_config(filename):
    r""" Merges specified config (or none) with default cfg file. """
    
    print(f" > Loading config ({filename})..", end='')
    
    if not filename:
        warnings.warn(f"(configs.py) WARNING: empty filename given.")
        experiment_cfg = {}
    else:
        if os.path.isfile(filename):
            cfg_file = filename
        else: 
            dir_path = pathlib.Path(__file__).parent.absolute()
            cfg_file = os.path.join(dir_path, filename)
            assert os.path.isfile(cfg_file), \
                f"{filename} not in configs ({dir_path})"
        with open(cfg_file, 'r') as f:
            experiment_cfg = yaml.safe_load(f)
    experiment_cfg = create_namespace(experiment_cfg)
    
    # Hyperparamter sampling
    hpsamp = None
    if 'hpsamp' in experiment_cfg.experiment:
        hpsamp = experiment_cfg.experiment.hpsamp
    if hpsamp:
        with open(hpsamp, 'r') as f:
            hpsamp_cfg = yaml.safe_load(f)
        experiment_cfg = get_sampled_config(hpsamp_cfg, experiment_cfg)
    print(f" done.")
    return experiment_cfg


# ============= ##   Nested Dict Namespace (Dot Notation)  ## ============= #


class NameSpace(dict):
    """ Enables dot notation for one layer in a nested dict config. """
    def __init__(self, *args, **kwargs):
        super(NameSpace, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_namespace(nested_dict):
    if not isinstance(nested_dict, dict):
        return nested_dict
    new_dict = {k: create_namespace(nested_dict[k]) for k in nested_dict}
    return NameSpace(new_dict)


# ============= ##  Config Sampling for Hyperparemter Tuning ## ============= #

SAMPLE_INSTRS = {'randint', 'randreal', 'sample'} 


def get_sampled_config(hparam_cfg, experiment_cfg):
    print(f"(HParam Sampling) Sampling values for hyperparameters in:")
    pprint.pprint(hparam_cfg)
    return nested_sample(hparam_cfg, experiment_cfg)


def nested_sample(hparam_cfg, cfg):
    for k, v in hparam_cfg.items():
        if isinstance(v, dict):
            cfg[k] = nested_sample(hparam_cfg[k], cfg[k])
        else:
            # print(k, v)
            instr = v
            sampled = sample(v)
            cfg[k] = sampled
            if isinstance(sampled, float):
                sampled = f'{sampled:.6f}'
            print(f"{k} = {sampled} ∼{v}.")
    return cfg


def sample(instr):
    
    def convert(val):
        if val.isdigit():
            return int(val)
        try:
            float(val)
        except ValueError:
            return val
        return float(val)

    if isinstance(instr, str):
        parts = instr.split('(')
        cmd = parts[0]
        if cmd in SAMPLE_INSTRS:
            import re
            attrs = re.findall(r"[0-9a-z.]+", parts[1])
            attrs = [convert(a) for a in attrs]
            if cmd == 'randint':
                assert len(attrs) == 2
                sampled = np.random.randint(int(attrs[0]), int(attrs[1]) + 1)
                return sampled
            elif cmd == 'randreal':
                assert len(attrs) == 2
                rand = np.random.rand()
                diff = float(attrs[1]) - float(attrs[0])
                sampled = float(attrs[0]) + rand * diff
                # print(f"Sampled {sampled} from '{instr}'.")
                return sampled
            elif cmd == 'sample':
                N = len(attrs)
                idx = np.random.randint(0, N)
                sampled = attrs[idx]
                # print(f"Sampled {sampled} from '{instr}'.")
                return sampled
        else:
            raise ValueError(f"Command {cmd} is not valid!")
    print(f"(Hparam Sample) WARNING: Instruction({instr}) is not valid.")
    return instr        


def _merge(default_d, experiment_d):
    merged_cfg = dict(default_d)
    for k, v in experiment_d.items():
        if isinstance(v, dict) and k in default_d:
            v = _merge(default_d[k], v)
        merged_cfg[k] = v
    return merged_cfg
