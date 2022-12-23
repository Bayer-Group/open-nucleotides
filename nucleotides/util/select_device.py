import os
import re

import torch
from gpustat.core import new_query
from pynvml import NVMLError

from nucleotides import config


def select_device(device_description: str = config.DEVICE, first: bool = True):
    """
    Get a device or list of devices based on input string
    Args:
        device_string: str describing preferred device
        first: just return one device, the first

    Returns: list of gpus

    """
    devices = []
    for device_string in device_description.split(","):
        if not device_string:
            continue
        if device_string == "auto":
            free_devices = get_free_devices()
            devices.append(free_devices[0])
        elif re.match("^auto:\d*$", device_string):
            free_devices = get_free_devices()
            n_devices = int(device_string[5:])
            devices.extend(free_devices[:n_devices])
        elif string_is_int(device_string):
            devices.append(torch.device(int(device_string)))
        else:
            devices.append(torch.device(device_string))
    if first:
        if not devices:
            return None
        return devices.pop()
    return devices


def get_free_devices(as_torch_device=True):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        info = new_query()
    except NVMLError:
        return []
    free_devices = [
        core.index for core in info if core.utilization == 0 and not core.processes
    ]
    if as_torch_device:
        free_devices = [torch.device(f"cuda:{index}") for index in free_devices]
    return free_devices


def string_is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


class NoFreeDevicevailableError(Exception):
    def __init__(self, device_descriptiom):
        super().__init__(
            f'No free devices "{device_descriptiom}" available at the moment'
        )


if __name__ == "__main__":
    print(select_device("auto:1,cpu"))
