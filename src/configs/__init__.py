
import os
import pathlib
import yaml
import pprint
import numpy as np


__all__ = ['get_config']

curr_path = pathlib.Path(__file__).parent.absolute()


class NameSpace(dict):
    """ Config file can be accessed as a dict or values as attributes. """
    def __init__(self, *args, **kwargs):
        super(NameSpace, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    @staticmethod
    def from_nested_dicts(dicts):
        if not isinstance(dicts, dict):
            return dicts
        else:
            return NameSpace({k: NameSpace.from_nested_dicts(dicts[k]) \
                              for k in dicts})



def get_config(filename):
    r""" Merges specified config (or none) with default cfg file. """
    
    print(f" > Loading config ({filename})..", end='')
    
    if not filename:
        print(f"(configs.py) WARNING: empty filename given.")
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
    experiment_cfg = NameSpace.from_nested_dicts(experiment_cfg)
    
    # Hyperparamter sampling
    hpsamp = experiment_cfg.experiment.hpsamp
    if hpsamp:
        with open(hpsamp, 'r') as f:
            hpsamp_cfg = yaml.safe_load(f)
        experiment_cfg = get_sampled_config(hpsamp_cfg, experiment_cfg)
    print(f" done.")
    return experiment_cfg


### ======================================================================== ###
### * ### * ### * ### *     Hyperparameter Sampling      * ### * ### * ### * ###
### ======================================================================== ###

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
