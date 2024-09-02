from typing import Tuple
import numpy as np
from device import Device
from arguments import get_args
import json

"""
    Information Links:
    1. https://www.nvidia.com/en-us/data-center/products/a10-gpu/
    2. https://www.nvidia.com/en-us/data-center/a100/
    3. https://shop.pegasus.hk/products/217290/NVIDIA-L20-GPU-Accelerator-PCIE-48GB      
    4. https://deepbaytech.com/images/nvidia-a800-datasheet-nvidia-a4-2521686-zhCN.pdf
    5. https://www.nvidia.com/en-us/data-center/h100/                 
    6. https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
"""

args = get_args()
with open(args.machine_config_path, 'r') as machine_config_file:
    machine_config = json.load(machine_config_file)

# Baseline
baseline_machines = machine_config['baseline_machines']

machine_specs = machine_config['machine_specs']

# sublist is in the same type, sublist has three number, indicating n_same_machine of 2,4,8 gpus
machine_amounts: dict = machine_config['machine_amounts']

ngpus = [2, 4, 8]

hetero_machines = []
used_gpus = 0
for name, machine_amount in machine_amounts.items():
    spec = machine_specs[name]
    for ngpu, n in machine_amount.items():
        if n == 0:
            continue
        used_gpus += int(ngpu)
        hetero_machines.append({"name": name, "tensor_core": spec[0], "memory": spec[1], "intra_bw": spec[2], "ngpus": int(ngpu), "n_same_machine":  n})


print(f"hetero machines: {hetero_machines}, ")
print(f"ngpus: {used_gpus}")
# assert used_gpus <= 128


baseline_mem = baseline_machines['memory'] * baseline_machines['ngpus'] * baseline_machines['n_same_machine']
baseline_tensor_core = baseline_machines['tensor_core'] * baseline_machines['ngpus'] * baseline_machines['n_same_machine']


hetero_mem = 0
hetero_tensor_core = 0

for machines in hetero_machines:
    hetero_mem += machines['memory'] * machines['ngpus'] * machines['n_same_machine']
    hetero_tensor_core += machines['tensor_core'] * machines['ngpus'] * machines['n_same_machine']

print(f"baseline mem: {baseline_mem}, hetero mem: {hetero_mem}, hetero-to-baseline mem ratio:\
      {hetero_mem / baseline_mem}, hetero mem over baseline: {hetero_mem > baseline_mem}")
print(f"baseline flops: {baseline_tensor_core}, hetero flops: {hetero_tensor_core}, hetero-to-baseline flops ratio:\
      {hetero_tensor_core / baseline_tensor_core}, hetero flops over baseline: {hetero_tensor_core > baseline_tensor_core}")

