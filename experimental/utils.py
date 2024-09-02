from device import Device
import json
from config import get_args


def create_machines_list():
    args = get_args()
    # hetero experiment
    with open(args.machine_config_path, 'r') as machine_config_file:
        machine_config = json.load(machine_config_file)
    machine_specs = machine_config['machine_specs']


    # sublist is in the same type, sublist has three number, indicating n_same_machine of 2,4,8 gpus
    # machine_amounts = np.random.randint(1, 3, size=(len(machine_specs), 3))
    machine_amounts = machine_config['machine_amounts']

    ngpus = [2, 4, 8]

    machines = []
    for name, machine_amount in machine_amounts.items():
        spec = machine_specs[name]
        for ngpu, n in machine_amount.items():
            if n == 0:
                continue
            machines.append({"name": name, "tensor_core": spec[0], "memory": spec[1], "intra_bw": spec[2], "ngpus": int(ngpu), "n_same_machine":  n})

    return machines


def create_specs(devices, inter_bw):
    
    tensor_cores = []   # (n, )
    for device in devices:
        tensor_cores.append(device.tensor_core * 1024*1024*1024*1024)

    comm_bws = []   # (n, n-1)
    comm_bws_dict = {}
    for i in range(len(devices)):
        comm_bw = []
        for j in range(len(devices)):
            if i != j:
                bw = inter_bw if devices[i].machine_id != devices[j].machine_id else devices[i].intra_bw 
                bw = bw * 1024*1024*1024
                comm_bw.append(bw)
                comm_bws_dict[i, j] = bw
        comm_bws.append(comm_bw)

    return tensor_cores, comm_bws, comm_bws_dict

def create_device_machine_map(devices):
    machine_ids = [d.machine_id for d in devices]
    return machine_ids
        

def create_devices(machines):
    devices = []

    assigned_id = 0
    for machine in machines:
        for i in range(machine['n_same_machine']):
            for j in range(machine['ngpus']):
                devices.append(Device(name=machine['name'], machine_id=assigned_id, 
                                  tensor_core=machine['tensor_core'], intra_bw=machine['intra_bw'],
                                  memory=machine['memory'], device_id=j + 1, machine_ngpus=machine['ngpus']))
            assigned_id += 1
    return devices