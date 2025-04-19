from jutils import jnn
import torch
from torch import nn

def test_lora():
    lin = nn.Linear(10, 20)
    lora_lin = jnn.LoRA(lin, r=4)
    x = torch.randn(16, 10)
    y = lin(x)
    y_lora = lora_lin(x)
    assert y == y_lora, "LoRA not outputting the same pred on init."

def test_loraify():
    resnet = Resnet(loaded_weights=True)
    loraify(resnet)
    assert resnet.modules() contains jnn.LoRA
    logs = train(resnet) on offtopic_dataset
    assert logs.loss decreases

