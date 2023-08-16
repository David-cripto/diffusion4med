from torch import nn
from torch import Tensor

class Unet(nn.Module):
    def __init__(self, channels = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.downpath = nn.ModuleList([])
        self.uppath = nn.ModuleList([])
        
        
    def forward(self, image: Tensor):
        pass