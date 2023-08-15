from torch import nn

class Unet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.downpath = nn.ModuleList([])
        self.uppath = nn.ModuleList([])
        
        