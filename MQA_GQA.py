import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "embed_size必须整除heads"

        self.values = torch.nn.Linear(embed_size, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(embed_size, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 最终的输出线性变换层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 查询分割成多头
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力
        energy = torch.einsum("nqhd,nkd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nld->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, attention


class GQA(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads):
        super(GQA, self).__init__()
        self.embed_dim = embed_dim
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.kv_head_dim = embed_dim // kv_heads
        self.query_head_dim = embed_dim // query_heads

        assert (
                self.kv_head_dim * kv_heads == embed_dim
        ), "embed_dim 必须整除 kv_heads"
        assert (
                self.query_head_dim * query_heads == embed_dim
        ), "embed_dim 必须整除 query_heads"

        self.values = nn.Linear(self.kv_head_dim, self.query_head_dim, bias=False)
        self.keys = nn.Linear(self.kv_head_dim, self.query_head_dim, bias=False)
        self.queries = nn.Linear(self.query_head_dim, self.query_head_dim, bias=False)

        # 将输出维度调整回原始嵌入维度
        self.fc_out = nn.Linear(kv_heads * self.kv_head_dim, embed_dim)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 查询分割成多头
        queries = queries.reshape(N, query_len, self.query_heads, self.query_head_dim)
        # 值和键分割成多头
        values = values.reshape(N, value_len, self.kv_heads, self.kv_head_dim)
        keys = keys.reshape(N, key_len, self.kv_heads, self.kv_head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        queries = queries.reshape(N, value_len, self.query_heads //self.kv_heads,self.kv_heads, self.query_head_dim)
        # 计算注意力
        energy = torch.einsum("nqphd,nkhd->nphqk", [queries, keys])
        GQA_attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=4)
        GQA_out = torch.einsum("nphqk,nkhd->nqphd", [GQA_attention, values]).reshape(
            N, query_len, self.query_heads * self.query_head_dim
        )

        GQA_out = self.fc_out(GQA_out)
        GQA_attention = GQA_attention.reshape(N, self.query_heads, query_len, key_len)
        return GQA_out, GQA_attention


# 参数定义
embed_size = 256
heads = 8
N = 1  # 批次大小
len = 10
query_heads = 8
kv_heads = 4

# 随机生成Q、K、V矩阵
x = torch.rand((N, len, embed_size))

# 创建多查询注意力实例
multi_query_attn = MultiQueryAttention(embed_size, heads)

# 创建查询组合注意力实例
gqa = GQA(embed_size, query_heads, kv_heads)

# 输出
out, attention = multi_query_attn(x, x, x)
print(out.shape, attention.shape)

GQA_out, GQA_attention = gqa(x, x, x)
print(GQA_out.shape, GQA_attention.shape)

# 绘制注意力权重
attention = attention.squeeze(0)  # 假设N=1，去掉批次维度
avg_attention = attention.mean(dim=0)  # 取头部的平均

# 绘制热力图
plt.figure(figsize=(10,8))
plt.imshow(avg_attention.cpu().detach().numpy(), cmap='viridis')
plt.title('MQAAttention Map')
plt.xlabel('Keys')
plt.ylabel('Queries')
plt.colorbar()
plt.show()

GQA_attention = GQA_attention.squeeze(0)  # 假设N=1，去掉批次维度
avg_GQA_attention = GQA_attention.mean(dim=0)  # 取头部的平均

# 绘制热力图
plt.figure(figsize=(10,8))
plt.imshow(avg_GQA_attention.cpu().detach().numpy(), cmap='viridis')
plt.title('GQA Attention Map')
plt.xlabel('Keys')
plt.ylabel('Queries')
plt.colorbar()
plt.show()