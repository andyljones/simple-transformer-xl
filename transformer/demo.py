import torch
from torch import nn
from torch.nn import functional as F
from . import nets
from tqdm.auto import tqdm

class Problem(nn.Module):
    """A simple, synthetic memory problem for demonstrating that the TransformerXL works.
    
    Each time you call `sample`, you get an `x, y` pair. The `x` is a random (T, B, 1) array filled 
    with (-1, 0, +1)s. The `y` is the sign of the rolling sum of `x`s over the last C steps. 
    
    If C is greater than 1, then to solve the task the net needs to access previous timesteps; if 
    C is greater than T, then it needs to remember previous chunks!

    Args:
        T: number of timesteps in each chunk
        C: number of timesteps the chunk is accumulated over
        B: batchsize
    """

    def __init__(self, T, C, B):
        super().__init__()
        self._T = T
        self._C = C
        self.register_buffer('cache', torch.zeros((C+T, B, 1)))
    
    def sample(self):
        B = self.cache.size(1)
        x = torch.randint(-1, +2, (self._T, B, 1), device=self.cache.device).float()
        self.cache[:] = torch.cat([self.cache, x], 0)[-self._C-self._T:]
        rolling = self.cache.cumsum(0) - self.cache.cumsum(0).roll(self._C, 0) 
        y = rolling.sign()
        return x, y[-self._T:]


class LSTMWrapper(nn.Module):
    """An alternative to the TransformerWrapper to demonstrate that an LSTM can't solve the memory problem
    when the relevant context is larger than the chunk size"""

    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.intake = nn.Linear(1, d_model)
        self.core = nn.LSTM(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def initialize(self, x):
        B = x.size(1)
        h = x.new_zeros((1, B, self.d_model))
        c = x.new_zeros((1, B, self.d_model))
        return (h, c)

    def forward(self, x, ms):
        x = F.relu(self.intake(x))
        x, (h, c) = self.core(x, ms)
        yhat = self.head(x)
        return yhat, (h.detach(), c.detach())


class TransformerWrapper(nets.TransformerXL):

    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, d_model=d_model, **kwargs)
        self.intake = nn.Linear(1, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, ms):
        x = F.relu(self.intake(x))
        h, ms = super().forward(x, ms)
        yhat = self.head(h)
        return yhat, ms


def train(T=3, C=6, B=1024, d_model=32):
    """Trains a Transformer XL to solve the multi-chunk rolling-memory `Problem` described above. 
    
    It should reach zero loss within a ~million samples, aka a few seconds on a 2019 GPU.

    If you swap the `TransformerWrapper` for an `LSTMWrapper`, it should have trouble getting
    past .05 loss.
    """
    problem = Problem(T, C, B).cuda()
    net = TransformerWrapper(d_model).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    x, y = problem.sample()
    ms = net.initialize(x)

    with tqdm() as pbar:
        while True:
            x, y = problem.sample()
            yhat, ms = net(x, ms)

            loss = F.mse_loss(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(T*B)
            pbar.set_postfix_str(f'Loss: {loss:.2f}')

if __name__ == '__main__':
    train()