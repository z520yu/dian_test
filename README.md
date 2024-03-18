## 学习记录

### 3.12

- **任务 1**: 实现问题一中的全部过程（不包括测试不同层数，神经元数）

代码如下：

[Neural Network.py](https://github.com/z520yu/dian_test/blob/master/Neural%20Network.py)

数据如下：

![accuracy.png](https://github.com/z520yu/dian_test/blob/master/accuracy.png)

![loss.png](https://github.com/z520yu/dian_test/blob/master/loss.png)


### 3.13

- **任务 1**: 实现任务二并进行可视化

- **学习内容**:了解RNN基本结构与计算方式，本质上是：将一张图片变成一个序列特征，每次将一个特征与上一次的隐藏层结合后算出结果和下一层隐藏层，通过用之前的输入特征来算隐藏层来达到让隐藏层记住之前特征的目的。

代码如下：

[RNN.py](https://github.com/z520yu/dian_test/blob/master/RNN.py)

数据如下：

![rnn_accuracy.png](https://github.com/z520yu/dian_test/blob/master/RNN_accuracy.png)

![rnn_loss.png](https://github.com/z520yu/dian_test/blob/master/rnn_loss.png)

### 3.15

- **任务 1**: 实现任务三的注意力权重计算与可视化

- **学习内容**:了解注意力机制基本原理：注意力本质是一种加权，通过将每一个token之间通过q,k矩阵计算权重获得不同token之间的关联程度，使词在进行翻译类似任务是可以关注到句子中的其他词，同时根据关联程度的高低，得到不同的权重，使翻译一个词是把注意力放到关联程度相对高的词上。

多头注意力计算方式如下图：
![multi_head.png](https://github.com/z520yu/dian_test/blob/master/multi_head.png)

自注意力计算如下图：

![calculate_method.png](https://github.com/z520yu/dian_test/blob/master/calculate_method.png)

代码如下：

[self-attention.py](https://github.com/z520yu/dian_test/blob/master/self-attention.py)

由于多头注意力，本代码为8个head，故只可视化第一个头的注意力权重，图片如下：


![attention hot map.png](https://github.com/z520yu/dian_test/blob/master/attention%20hot%20map.png)

### 3.16

- **任务 1**: 修正昨天多头注意力代码。

- **学习内容**:笔者在学习MQA,GQA时，重新阅读了一下多头注意力机制的计算方式，发现网上有两种说法，一种是每个序列完整的给到单个头，算出多个头的结果后线性变换，另一种是将一个序列切割成八份（八个头）分别给到不同的头进行计算，最后再合并后线性变换。我便找到论文，阅读后发现应该是后者正确，我昨天的代码是前者的做法，故今天修改。

论文中多头注意力解释如下图：
![multi_head_paper.png](https://github.com/z520yu/dian_test/blob/master/multi_head_paper.png)


修改后代码如下：

[multi_head_attention.py](https://github.com/z520yu/dian_test/blob/master/multi_head_attention.py)

这次我将八个头的注意力取平均后画成热力图，图片如下：


![ave_attention_map.png](https://github.com/z520yu/dian_test/blob/master/ave_attention_map.png)

- **任务 2**: 完成MQA,GQA注意力代码。

- **学习内容**:学习MQA，GAQ的基本原理，MQA就是多个查询，一个键，一个值，就把查询按注意力头数切割，键和值线性变换成查询的大小，然后在每个头分别计算后再剪切到一起后线性变换得到输出；GAQ即按查询分组，可以理解成，查询的头数与键，值头数成倍数，一个键值对应多个查询，我这里将查询按照键，值的头数增加张量维数，即2*4（查询头数），4（键，值头数），便于进行张量乘法计算。


代码如下：

[MQA_GQA.py](https://github.com/z520yu/dian_test/blob/master/MQA_GQA.py)

也是将八个头的注意力取平均后画成热力图，MQA与GQA热力图如下：


![MQA_attention.png](https://github.com/z520yu/dian_test/blob/master/MQA_attention.png)

![GQA_attention.png](https://github.com/z520yu/dian_test/blob/master/GQA_attention.png)


### 3.17

- **任务 1**: 使用NNI实现自动化机器学习调节超参。

- **学习内容**:阅读NNI官方文档，学习如何使用NNI框架为模型选择合理参数。
本次调节参数的搜索空间如下：
```yaml
search_space:
  features1:
    _type: 'choice'
    _value:
      - 128
      - 256
      - 512
      - 1024
  features2:
    _type: 'choice'
    _value:
      - 128
      - 256
      - 512
      - 1024
  lr:
    _type: 'loguniform'
    _value:
      - 0.0001
      - 0.1
  batch_size:
    _type: 'choice'
    _value:
      - 32
      - 64
      - 128
      - 256
```

使用TPE即贝叶斯优化来进行搜索，一共尝试了10次。


因需要使用框架，对第一题的代码进行略微修改后代码如下：

[nn_choose_params.py](https://github.com/z520yu/dian_test/blob/master/nn_choose_params.py)

框架调用代码如下：

[trial.py](https://github.com/z520yu/dian_test/blob/master/trial.py)

10次测试的结果图片如下：


![params_result.png](https://github.com/z520yu/dian_test/blob/master/params_result.png)


可以看到，最高的accuracy为0.979，其各参数设置如下：

```json
{
    "features1": 512,
    "features2": 1024,
    "lr": 0.05845589437848884,
    "batch_size": 32
}
```

可能是20个epoch，大多数都没有过拟合，通过我的观察，学习率是影响相对最大的，其他的几个参数也都有影响。





### 3.18

- **任务 1**: 对NN进行层数测试。

- **学习内容**:学习了NNI框架下的NAS,即模型架构探索，并对最初的Neural Network.py进行改编。

代码如下：

[Neural_Architecture_Search.py](https://github.com/z520yu/dian_test/blob/master/Neural_Architecture_Search.py)

这里将对六种中间层数进行研究，一层的为256，两层的为128，256，三层的为128，256，256，后续三个都是加了一个256。

事实上，单纯实现连续加一层256神经元的全连接层，用一个for循环就能实现，但笔者认为，后续的探索模型架构肯定不是简单的加一层两层，故使用该框架进行最优层数的搜索,考虑到训练时间，故只放了六种。

(ps:此框架近一两年才出现，支持环境不多，我在实施的过程中遇见了很多问题，均在github的issues中可以找到，但大多没有人给出解答。)

数据图片如下：

![nas_result.png](https://github.com/z520yu/dian_test/blob/master/nas_result.png)

图表展示如下：

|  层数 | 准确率  | 时间   |
|------|---------|--------|
| 5    | 0.982285| 23m15s |
| 4    | 0.978901| 7m55s  |
| 6    | 0.978404| 14m    |
| 2    | 0.978304| 4m25s  |
| 3    | 0.978304| 5m3s   |
| 1    | 0.976612| 4m18s  |

可以看到，层数变多，大概率会导致训练时间变长，但层数变多并不一定正确率就高，可能5层就是对于这个问题相对好的，层数再多，到后面层就丢失信息了。

（ps:由于这个框架要求数据序列化，不能太大，所以这次的数据是进行标准化了的）
