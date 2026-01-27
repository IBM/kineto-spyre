import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import platform

from torch_sendnn import torch_sendnn  # noqa: F401

from torch.profiler import profile, record_function, ProfilerActivity

if platform.machine() == "s390x":
    from torch.serialization import LoadEndianness
    torch.serialization.set_default_load_endianness(LoadEndianness.LITTLE)

torch.utils.rename_privateuse1_backend("aiu")
torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
torch.utils.generate_methods_for_privateuse1_backend()

torch.manual_seed(0xAFFE)

class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(32, 32)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

model = TinyModel()

with torch.no_grad():
#    compiled_model = torch.compile(model )
    compiled_model = torch.compile(model, backend="sendnn")

    x = torch.randn(32, 32)
    out = ""
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], 
                 record_shapes=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/matmul')) as prof:
            out = compiled_model(x)
    print('======================== x ====================')
    print(x)
    print('======================= out ===================')
    print(out)
    print('''
======================= cpu ===================
tensor([[0.0000, 0.9447, 1.4019,  ..., 0.0000, 0.1016, 0.0000],
        [0.0173, 1.2010, 1.0872,  ..., 0.4546, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.4439,  ..., 0.7030, 0.8734, 0.3726],
        ...,
        [0.0441, 0.0554, 0.0000,  ..., 0.0000, 1.3401, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.4113, 0.5703],
        [0.7894, 0.2620, 0.0000,  ..., 0.0000, 0.2320, 0.8736]])''')
    
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10).replace("CUDA", "AIU"))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10).replace("CUDA", "AIU"))