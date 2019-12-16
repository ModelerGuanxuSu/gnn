*编辑：苏冠旭 2019/12/3*

<p align="center">
    <a href="https://github.com/python/cpython"><img src="https://img.shields.io/badge/Python-3.7-FF1493.svg"></a>
    <a href="https://github.com/tensorflow/tensorflow"><img src="https://img.shields.io/badge/TensorFlow-1.14.0-blue"></a>
    <a href="https://opensource.org/licenses/mit-license.php"><img src="https://badges.frapsoft.com/os/mit/mit.svg"></a>
</p>

## 版本要求

- **Tensorflow** 1.15.0 或1.14.0
- **Python** 3.7


## 训练过程

- **1. 定义传播矩阵A**：

    传播矩阵A可以是GCN定义的方式，也可以是用户自定义的其他传播矩阵。
A的第**i**行第**j**列为节点**j**到节点**i**的传播权重
    
    矩阵A的数据类型为稀疏矩阵模块scipy.sparse下
的scipy.sparse.coo.coo_matrix。如果用户自定义的A为scipy.sparse下的
其他类型的稀疏矩阵，可以通过.tocoo()方法转换为所要求的数据类型
    
- **2. 模型训练**：

    需要注意的是，图神经网络在训练和预测的时候都需要将全图数据传入，
以便风险特征在传播过程中没有信息损失。
    
    因此，X为训练集和测试集通用的，X中甚至还包含训练样本之外的样本。
比如训练样本为新金融企业，X需要含有所训练新金融企业的关联方，这些关联方
并不在训练样本范围内，但是需要包含在X中，并体现在传播矩阵A上

    Y的长度与X的长度相同。对于非训练样本的Y，可以设置为任意值
    
    鉴于以上原因。与通常习惯不同的是，训练和预测的时候，我们需要传入
相同的X和Y。在训练时，我们通过train_mask来区分训练数据。算法内部只会
对train_mask大于0的位置对应的X和Y进行训练。

- **2. 模型预测**：

    在预测时，不但需要使得X包含所需要预测的节点，还应包含该节点的k度以内
的邻居节点。A矩阵同理。得到预测值之后，我们对预测值进行切片，拿到我们
想要预测的节点预测值即可。