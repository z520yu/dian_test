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



