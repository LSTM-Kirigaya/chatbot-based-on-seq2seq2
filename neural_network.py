# net.py
import torch
import torch.nn as nn

# 前馈的encoder
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1,dropout=0):
        """
        :param hidden_size:  RNN隐藏层的维度，同时也是词向量的维度
        :param embedding:  外部的词向量嵌入矩阵
        :param n_layers: 单个RNNcell的层数
        :param dropout: 单个RNN过程隐层参数被丢弃的概率，如果n_layer==1，那么这个参数设为0
        """
        super(EncoderRNN, self).__init__()

        self.n_layer = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 此处词向量维度（嵌入维度）和RNN隐藏层维度都是hidden_size
        # 下面定义GRU算子，其中bidirectional=True代表我们的GRU会对序列双向计算
        # 我们按照batch_first的数据输入维度来定义gru
        self.gru = nn.GRU(input_size=hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=n_layers,
                                        dropout=(0 if n_layers==1 else dropout),
                                        bidirectional=True,
                                        batch_first=True)

    # 定义前向逻辑
    def forward(self, input_seq, input_length, hidden=None):
        """
        :param input_seq:  一个batch的index序列
        :param input_length:  对应每个index序列长度的tensor
        :param hidden:  三阶张量，代表RNN的第一个隐藏层(h0)的初值
        :return: [output, hidden]. 其中output为RNN的所有输出ot，hidden为RNN最后一个隐藏层
        """
        # 首先将index序列转化为词向量组成的矩阵
        embedded = self.embedding(input_seq) # embedded:[batch_size, max_length, embedding_dim], 此处embedding_dim==hidden_size

        # 为了避免让RNN迭代过程中计算PAD的词向量(它理应是全0向量)的那一行
        # 使用pack_padded_sequence方法进行压缩；
        # 压缩会按照传入的句长信息，只保留句长之内的词向量
        # 压缩会返回一个PackedSequence类，GRU算子可以迭代该数据类型
        # GRU算子迭代完后返回的还是PackedSequence类，届时我们需要使用pad_packed_sequence，将PackedSequence类解压为tensor

        # 压缩
        packed = nn.utils.rnn.pack_padded_sequence(input=embedded,
                                                                               lengths=input_length,
                                                                               batch_first=True)
        # 前向传播, 使用hidden初始化h_0
        outputs, hidden = self.gru(packed, hidden)
        # 由于hidden的shape不受输入句长的影响，所以outputs还是PackedSequence类，但hidden已经自动转换成tensor类了

        # 解压outputs， 返回解压后的tensor和原本的长度信息
        outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=outputs,
                                                                                  batch_first=True,
                                                                                  padding_value=0)

        # outputs: [batch_size, max_length, 2 * hidden_size]
        # 将双向RNN的outputs按位加起来
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size : ]
        # outputs: [batch_size, max_length, hidden_size]
        # hidden: [batch_size, 2, hidden_size]
        return outputs, hidden

# Luong提出的全局注意力机制层
class LuongAttentionLayer(nn.Module):
    def __init__(self, score_name, hidden_size):
        """
        :param score_name: score函数的类型
        :param hidden_size: 隐层的维度
        :return:
        """
        # score函数是Luong论文中提出用来衡量decoder的目前的隐层和encoder所有输出的相关性
        # 论文中不仅介绍了score函数的概念，而且给出了三种映射：dot,general,concat
        # 这三种score函数都是向量与向量到标量的映射
        super(LuongAttentionLayer, self).__init__()
        self.score_name = score_name
        self.hidden_size = hidden_size

        if score_name not in ["dot", "general", "concat"]:
            raise ValueError(self.score_name, "is not an appropriate attention score_name(dot, general, concat)")
        # general有一个向量的外积变换， concat有一个外积变换与一个内积变换，所以这两个方法会产生额外的网络参数
        # 下面为这两个方法设置需要的网络参数
        if score_name == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size)   # 使用h->h的全连接层模拟h*h的矩阵与h维列向量的乘法
        elif score_name == "concat":
            self.Wa = nn.Linear(hidden_size * 2, hidden_size)  # 需要将h_t与h_s联级再做矩阵乘法，所以是2h->h的全连接层
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))  # FloatTensor会生成hidden_size维的全零（浮点0）向量，我们通过参数可更新的向量来模拟内积变换

        # 计算attention vector的变换
        self.Wc = nn.Linear(hidden_size * 2, hidden_size)   # 需要将c_t与h_t联级再做矩阵乘法，所以是2h->h的全连接层

    # 下面定义三个score：
    # 其中： current_hidden : [batch_size, 1, hidden_size]
    #          encoder_output : [batch_size, max_length, hidden_size]
    def dot_score(self, current_hidden, encoder_ouput):
        return torch.sum(encoder_ouput * current_hidden, dim=2)

    def general_score(self, current_hidden, encoder_output):
        energy = self.Wa(encoder_output)
        # energy : [batch_size, max_length, hidden_size]
        return torch.sum(energy * current_hidden, dim=2)

    def concat_score(self, current_hidden, encoder_output):
        concat = torch.cat([current_hidden.expand(encoder_output.shape), encoder_output], dim=2)
        # concat : [batch, max_length, 2 * hidden_size]
        energy = self.Wa(concat).tanh()
        return torch.sum(self.v * energy, dim=2)

    # 前馈运算为encoder的全体output与decoder的current output值到attention向量的映射
    def forward(self, current_hidden, encoder_output):
        if self.score_name == "dot":
            score = self.dot_score(current_hidden, encoder_output)
        elif self.score_name == "general":
            score = self.general_score(current_hidden, encoder_output)
        elif self.score_name == "concat":
            score = self.concat_score(current_hidden, encoder_output)
        # score : [batch_size, max_length]
        # 通过 softmax将score转化成注意力权重alpha
        attention_weights = nn.functional.softmax(score, dim=1)
        # attention_weights : [batch_size, max_length]
        # 通过encoder_output与alpha的加权和得到context vector(上下文向量)
        # 通过广播运算完成
        context_vector = torch.sum(encoder_output * attention_weights.unsqueeze(2), dim=1)
        # context_vector : [batch_size, hidden_size]
        # 有了上下文向量，就可以计算attention vector了
        attention_vector = self.Wc(torch.cat([context_vector.unsqueeze(1), current_hidden], dim=2)).tanh()
        # attention_vector : [batch_size, 1, hidden_size]
        return attention_vector

# decoder
class LuongAttentionDecoderRNN(nn.Module):
    def __init__(self, score_name, embedding, hidden_size, output_size, n_layers=1, dropout=0):
        """
        :param score_name:  Luong Attention Layer 的score函数的类型，可选：dot, general, concat
        :param embedding:  词向量嵌入矩阵
        :param hidden_size:  RNN隐层维度
        :param output_size:  每一个cell的输出维度，一般会通过一个 hidden_size->output_size的映射把attention vector映射到output_size维向量
        :param n_layers:  单个RNN cell的层数
        :param dropout:  代表某层参数被丢弃的概率，在n_layers==1的情况下设为0
        """
        super(LuongAttentionDecoderRNN, self).__init__()

        self.score_name = score_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义或获取前向传播需要的算子
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=n_layers,
                                     dropout=(0 if n_layers == 1 else dropout),
                                     batch_first=True)
        self.attention = LuongAttentionLayer(score_name, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    # decoder的前向传播逻辑
    # 需要注意的是，不同一般的RNN，由于我们的decoder输出需要做注意力运算
    # 所以decoder的RNN需要拆开来算，也就是我们的forward一次只能算一个time_step，而不是一次性循环到底
    def forward(self, input_single_seq, last_hidden, encoder_output):
        """
        :param input_single_seq:  上一步预测的单词对应的序列  [batch_size, 1]
        :param last_hidden:  decoder上次RNN 的隐层           [batch_size, 2, hidden_size]
        :param encoder_output:  encoder的全体输出        [batch_size, max_length, hidden_size]
        :return: 代表个位置概率的output向量[batch_size, output_size]和current hidden [batch_size, 2, hidden_size]
        """

        embedded = self.embedding(input_single_seq)   # [batch_size, max_length, hidden_size]
        embedded = self.embedding_dropout(embedded)     # [batch_size, max_length, hidden_size]

        current_output, current_hidden = self.gru(embedded, last_hidden)
        # current_output : [batch_size, 1, hidden_size]
        # current_hidden : [1, batch_size, hidden_size]

        # 这里拿current_output当做current_hidden来参与attention运算
        # 当然，直接拿current_hidden也行
        attention_vector = self.attention(current_output, encoder_output)
        # attention_vector : [batch_size, 1, hidden_size]

        # 将attention_vector变到output_size维
        output = self.hidden2output(attention_vector)
        # output : [batch_size, 1, output_size]
        output = nn.functional.softmax(output.squeeze(1), dim=1)
        # output : [batch_size, output_size]
        return output, current_hidden


if __name__ == "__main__":
    embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
    encoder = EncoderRNN(hidden_size=16, embedding=embedding, n_layer=1, dropout=0)

    decoder = LuongAttentionDecoderRNN(score_name="dot", embedding=embedding, hidden_size=16, output_size=10, n_layers=1, dropout=0)

    x = torch.tensor([[2], [2]])
    out = torch.arange(320).reshape([2, 10, 16])
    length = torch.tensor([5, 3])

    #outputs, hidden = encoder.forward(input_seq=x, input_length=length)

    decoder.forward(input_single_seq=x, last_hidden=None, encoder_output=out)
