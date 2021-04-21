from torch import nn
from attention import FeedForward, RectifiedLinearAttention, PreNorm


class Transformer(nn.Module):

    def __init__(self, depth, dim, heads, dim_head, scale, dropout):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, RectifiedLinearAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                                          rmsnorm=True)),
                    PreNorm(dim, FeedForward(dim, dim*scale, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x