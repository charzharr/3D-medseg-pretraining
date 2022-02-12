
import os
import psutil
import subprocess
import torch


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    
    return gpu_memory_map


def ram(disp=False):
    """ Return (opt display) RAM usage of current process in megabytes. """
    process = psutil.Process(os.getpid())
    bytes = process.memory_info().rss
    mbytes = bytes // 1048576
    sys_mbytes = psutil.virtual_memory().total // 1048576
    if disp:
        print(f'🖥️  Current process (id={os.getpid()}) '
              f'RAM Usage: {mbytes:,} MBs / {sys_mbytes:,} Total MBs.')
    return mbytes


def mem(gpu_indices):
    """ Get primary GPU card memory usage in gigabytes. """
    if not torch.cuda.is_available() or not gpu_indices:
        return -1
    mem_map = devices.get_gpu_memory_map()
    prim_card_num = int(gpu_indices[0])
    return mem_map[prim_card_num] / 1024

