import third_party.megatron.core.parallel_state
import third_party.megatron.core.tensor_parallel
import third_party.megatron.core.utils

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
]
