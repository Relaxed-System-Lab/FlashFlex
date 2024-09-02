class Device:
    def __init__(self, name, machine_id, tensor_core, intra_bw, memory, **kwargs):
        self.name = name
        self.machine_id = machine_id
        self.tensor_core = tensor_core
        self.intra_bw = intra_bw
        self.memory = memory
        self.kwargs = kwargs
    
    def __str__(self):
        return f"{self.name}, {self.kwargs['device_id']} / {self.kwargs['machine_ngpus']}, tensor_core={self.tensor_core}, memory={self.memory}"
