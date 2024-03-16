import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "embed_size必须整除heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 最终的输出线性变换层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 分割成多头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, attention


# 参数定义
embed_size = 256
heads = 8
N = 1  # 批次大小
seq_len = 10  # 序列长度


# 随机生成Q、K、V矩阵
x = torch.rand((N, seq_len, embed_size))

# 创建多头注意力实例
multi_head_attn = MultiHeadAttention(embed_size, heads)


# 计算输出
output, attention = multi_head_attn(x, x, x)

print(output)  # torch.Size([1, 10, 256])

attention = attention.squeeze(0)  # 假设N=1，去掉批次维度
avg_attention = attention.mean(dim=0)  # 取头部的平均

# 绘制热力图
plt.figure(figsize=(10,8))
plt.imshow(avg_attention.cpu().detach().numpy(), cmap='viridis')

plt.title('Attention Map')
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')
plt.colorbar()
plt.show()



