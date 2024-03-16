from torch.nn import Module, Linear, ReLU

class FeedForward(Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()

        self.layer_1 = Linear(d_model, d_ff)
        self.layer_2 = Linear(d_ff, d_model)

        self.relu = ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(out)
        
        x = self.layer_2(x)

        return out
