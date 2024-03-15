import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class SimplifiedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SimplifiedMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        # 对每个头使用不同的线性变换矩阵生成Qi、Ki、Vi
        self.values = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(heads)])
        self.keys = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(heads)])
        self.queries = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(heads)])

        # 最终的输出线性变换层
        self.fc_out = nn.Linear(heads * embed_size, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 存储每个头的输出
        output_heads = []
        attention_weights_list = []  # 用于保存每个头的注意力权重

        # 对每个头分别计算注意力机制
        for head in range(self.heads):
            value = self.values[head](values) # 括号中的values是输入的value矩阵
            key = self.keys[head](keys)
            query = self.queries[head](queries)

            energy = torch.einsum("nqd,nkd->nqk", [query, key])
            attention = F.softmax(energy / (self.embed_size ** 0.5), dim=2)
            attention_weights_list.append(attention)
            head_output = torch.einsum("nqk,nkd->nqd", [attention, value])
            output_heads.append(head_output)

        # 将所有头的输出拼接，并通过最终的线性层
        out = torch.cat(output_heads, dim=2)
        out = self.fc_out(out)

        return out , attention_weights_list


# 参数定义
embed_size = 16
heads = 8
N = 1  # 批次大小
seq_len = 10  # 序列长度


# 随机生成Q、K、V矩阵
x = torch.rand((N, seq_len, embed_size))

# 创建多头注意力实例
simplified_multi_head_attn = SimplifiedMultiHeadAttention(embed_size, heads)


# 计算输出
output, attention_weights = simplified_multi_head_attn(x, x, x)

print(output)  # torch.Size([1, 10, 16])

attention_first_head_weights = attention_weights[0].detach().cpu().numpy()

# 将attention_weights的第一个头的注意力权重转换为numpy数组
attention_first_head_weights = attention_first_head_weights.reshape(seq_len, seq_len)

# 绘制热图
plt.figure(figsize=(10, 8))
plt.imshow(attention_first_head_weights, cmap='viridis')
plt.colorbar()
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')
plt.title('Attention Heatmap')
plt.show()

