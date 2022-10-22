from itertools import zip_longest
import random
import argparse
import os
import math

import torch
import torch.nn as nn

PAD_token = 0  # 补足句长的pad占位符的index
SOS_token = 1  # 代表一句话开头的占位符的index
EOS_token = 2  # 代表一句话结尾的占位符的index
UNK_token = 3  # 代表不在词典中的字符


# 用来构造字典的类
class vocab(object):

    def __init__(self, name, pad_token, sos_token, eos_token, unk_token):
        self.name = name
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.trimmed = False  # 代表这个词表对象是否经过了剪枝操作
        self.word2index = {
            "PAD": pad_token,
            "SOS": sos_token,
            "EOS": eos_token,
            "UNK": unk_token
        }
        self.word2count = {"UNK": 0}
        self.index2word = {
            pad_token: "PAD",
            sos_token: "SOS",
            eos_token: "EOS",
            unk_token: "UNK"
        }
        self.num_words = 4  # 刚开始的四个占位符 pad(0), sos(1), eos(2)，unk(3) 代表目前遇到的不同的单词数量

    # 向voc中添加一个单词的逻辑
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 向voc中添加一个句子的逻辑
    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    # 将词典中词频过低的单词替换为unk_token
    # 需要一个代表修剪阈值的参数min_count，词频低于这个参数的单词会被替换为unk_token，相应的词典变量也会做出相应的改变
    def trim(self, min_count):
        if self.trimmed:  # 如果已经裁剪过了，那就直接返回
            return
        self.trimmed = True

        keep_words = []
        keep_num = 0
        for word, count in self.word2count.items():
            if count >= min_count:
                keep_num += 1
                # 由于后面是通过对keep_word列表中的数据逐一统计，所以需要对count>1的单词重复填入
                for _ in range(count):
                    keep_words.append(word)

        print("keep words: {} / {} = {:.4f}".format(
            keep_num, self.num_words - 4, keep_num / (self.num_words - 4)))

        # 重构词表
        self.word2index = {
            "PAD": self.pad_token,
            "SOS": self.sos_token,
            "EOS": self.eos_token,
            "UNK": self.unk_token
        }
        self.word2count = {}
        self.index2word = {
            self.pad_token: "PAD",
            self.sos_token: "SOS",
            self.eos_token: "EOS",
            self.unk_token: "UNK"
        }
        self.num_words = 4

        for word in keep_words:
            self.addWord(word)

    # 读入数据，统计词频，并返回数据
    def load_data(self, path):
        pairs = []
        for line in open(path, "r", encoding="utf-8"):
            try:
                input_dialog, output_dialog = line.strip().split("\t")
                self.addSentence(input_dialog.strip())
                self.addSentence(output_dialog.strip())
                pairs.append([input_dialog, output_dialog])
            except:
                pass
        return pairs


# 处理数据，因为我们所有低词频和无法识别的单词（就是不在字典中的词）都替换成UNK
# 但是我们不想要看到程序输出给用户的话中带有UNK，所以下面做的处理为：
# 先判断output_dialog，如果其中含有不存在与字典中的词，则直接舍弃整个dialog，
# 如果output_dialog满足要求，那么我们将input_dialog中无法识别的单词全部替换为UNK
def trimAndReplace(voc, pairs, min_count):
    keep_pairs = []
    # 先在词表中删去低频词
    voc.trim(min_count)
    for input_dialog, output_dialog in pairs:
        drop_dialog = False
        # 判断output_dialog中是否有无法识别的词， 有的话，直接跳过这个循环
        for word in output_dialog.split():
            if word not in voc.word2index:
                drop_dialog = True
        if drop_dialog:
            continue

        # 输入句的未识别单词用UNK替换
        input_dialog = input_dialog.split()
        for idx, word in enumerate(input_dialog):
            if word not in voc.word2index:
                input_dialog[idx] = "UNK"
        input_dialog = " ".join(input_dialog)

        keep_pairs.append([input_dialog, output_dialog])

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs),
        len(keep_pairs) / len(pairs)))
    return keep_pairs


# 将一句话转换成id序列(str->list)，结尾加上EOS
def sentenceToIndex(sentence, voc):
    return [voc.word2index[word]
            for word in sentence.split()] + [voc.eos_token]


# 使用mask loss的方法避免计算pad的loss
# 计算非pad对应位置的交叉熵（负对数似然）
def maskNLLLoss(output, target, mask, device):
    """
    :param output: decoder的所有output的拼接    [batch_size, max_length, output_size]
    :param target: 标签，也就是batch的output_dialog对应的id序列  [batch_size, max_length]
    :param mask: mask矩阵    与target同形
    :return: 交叉熵损失、单词个数
    """

    target = target.type(torch.int64).to(device)
    mask = mask.type(torch.BoolTensor).to(device)

    total_word = mask.sum()  # 单词个数
    crossEntropy = -torch.log(
        torch.gather(output, dim=2, index=target.unsqueeze(2)))
    # crossEntropy : [batch_size, max_length, 1]
    loss = crossEntropy.squeeze(2).masked_select(mask).mean()
    loss = loss.to(device)
    return loss, total_word.item()


# Leslie的Triangle2学习率衰减方法
def Triangular2(T_max, gamma):

    def new_lr(step):
        region = step // T_max + 1  # 所处分段
        increase_rate = 1 / T_max * math.pow(gamma, region - 2)  # 增长率的绝对值
        return increase_rate * (step - (region - 1) * T_max) if step <= (
            region - 0.5) * T_max else -increase_rate * (step - region * T_max)

    return new_lr


# 将一个batch中的input_dialog转化为有pad填充的tensor，并返回tensor和记录长度的变量
# 返回的tensor是batch_first的
def batchInput2paddedTensor(batch, voc):
    # 先转换为id序列，但是这个id序列不对齐
    batch_index_seqs = [sentenceToIndex(sentence, voc) for sentence in batch]
    length_tensor = torch.tensor(
        [len(index_seq) for index_seq in batch_index_seqs])
    # 下面填充0(PAD)，使得这个batch中的序列对齐
    zipped_list = list(zip_longest(*batch_index_seqs, fillvalue=voc.pad_token))
    padded_tensor = torch.tensor(zipped_list).t()
    return padded_tensor, length_tensor


# 将一个batch中的output_dialog转化为有pad填充的tensor，并返回tensor、mask和最大句长
# 返回的tensor是batch_first的
def batchOutput2paddedTensor(batch, voc):
    # 先转换为id序列，但是这个id序列不对齐
    batch_index_seqs = [sentenceToIndex(sentence, voc) for sentence in batch]
    max_length = max([len(index_seq) for index_seq in batch_index_seqs])
    # 下面填充0(PAD)，使得这个batch中的序列对齐
    zipped_list = list(zip_longest(*batch_index_seqs, fillvalue=voc.pad_token))
    padded_tensor = torch.tensor(zipped_list).t()
    # 得到padded_tensor对应的mask
    mask = torch.BoolTensor(zipped_list).t()
    return padded_tensor, mask, max_length


def train_one_batch(input_seq,
                    input_length,
                    device,
                    target,
                    mask,
                    max_target_len,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    batch_size,
                    clip,
                    teacher_forcing_ratio,
                    encoder_lr_scheduler=None,
                    decoder_lr_scheduler=None):
    """
        :param input_seq:  输入encoder的index序列                                         [batch_size, max_length]
        :param input_length:  input_seq中每个序列的长度                                [batch_size]
        :param target:  每个input_dialog对应的output_dialog的index序列      [batch_size, max_length]
        :param mask:  target对应的mask矩阵                                                    [batch_size, max_length]
        :param max_target_len:  target中的最大句长
        :param encoder: encoder实例
        :param decoder: decoder实例
        :param encoder_optimizer: 承载encoder参数的优化器
        :param decoder_optimizer: 承载decoder参数的优化器
        :param batch_size: batch大小
        :param clip: 修剪梯度的阈值，超过该值的导数值会被裁剪
        :param teacher_forcing_ratio: 训练该batch启动teacher force的策略
        :param encoder_lr_scheduler: encoder_optimizer学习率调整策略
        :param decoder_lr_scheduler: decoder_optimizer学习率调整策略
        :return: 平均似然损失
        """
    # 清空优化器
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 设置变量的运算设备
    input_seq = input_seq.to(device)
    input_length = input_length.to(device)
    target = target.to(device)
    mask = mask.to(device)

    decoder_output = torch.FloatTensor()  # 用来保存decoder的所有输出
    decoder_output = decoder_output.to(device)

    # encoder的前向计算
    encoder_output, encoder_hidden = encoder(input_seq, input_length)

    # 为decoder的计算初始化一个开头SOS
    decoder_input = torch.tensor([SOS_token
                                  for _ in range(batch_size)]).reshape([-1, 1])
    decoder_input = decoder_input.to(device)

    # 根据decoder的cell层数截取encoder的h_n的层数
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 根据概率决定是否使用teacher force
    use_teacher_forcing = True if random.random(
    ) < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for i in range(max_target_len):
            # 执行一步decoder前馈，只计算了一个单词
            output, current_hidden = decoder(decoder_input, decoder_hidden,
                                             encoder_output)
            # output : [batch_size, output_size]
            # current_hidden : [1, batch_size, hidden_size]
            # 将本次time_step得到的结果放入decoder_output中
            decoder_output = torch.cat(
                [decoder_output, output.unsqueeze(1)], dim=1)
            # 使用teacher forcing：使用target（真实的单词）作为decoder的下一次的输入，而不是我们计算出的预测值（计算出的概率最大的单词）
            decoder_input = target[:, i].reshape([-1, 1])
            decoder_hidden = current_hidden
    else:
        for i in range(max_target_len):
            # 执行一步decoder前馈，只计算了一个单词
            output, current_hidden = decoder(decoder_input, decoder_hidden,
                                             encoder_output)
            # output : [batch_size, output_size]
            # current_hidden : [1, batch_size, hidden_size]
            # 从softmax的结果中得到预测的index
            predict_index = torch.argmax(output, dim=1)
            # 将本次time_step得到的结果放入decoder_output中
            decoder_output = torch.cat(
                [decoder_output, output.unsqueeze(1)], dim=1)
            decoder_input = predict_index.reshape([-1, 1])
            decoder_hidden = current_hidden

    # 计算本次batch中的mask_loss和总共的单词数
    mask_loss, word_num = maskNLLLoss(decoder_output, target, mask, device)

    # 开始反向传播，更新参数
    mask_loss.backward()

    # 裁剪梯度
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 更新网络参数
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 调整优化器的学习率
    if encoder_lr_scheduler and decoder_lr_scheduler:
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()

    return mask_loss.item()


# 获取数据加载器的函数
# 将输入的一个batch的dialog转换成id序列，填充pad，并返回训练可用的id张量和mask
def DataLoader(pairs, voc, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(pairs)

    batch = []
    for idx, pair in enumerate(pairs):
        batch.append([pair[0], pair[1]])

        # 数据数量到达batch_size就yield出去并清空
        if len(batch) == batch_size:
            # 为了后续的pack_padded_sequence操作，我们需要给这个batch中的数据按照input_dialog的长度排序(降序)
            batch.sort(key=lambda x: len(x[0].split()), reverse=True)
            input_dialog_batch = []
            output_dialog_batch = []
            for pair in batch:
                input_dialog_batch.append(pair[0])
                output_dialog_batch.append(pair[1])

            input_tensor, input_length_tensor = batchInput2paddedTensor(
                input_dialog_batch, voc)
            output_tensor, mask, max_length = batchOutput2paddedTensor(
                output_dialog_batch, voc)

            # 清空临时缓冲区
            batch = []

            yield [
                input_tensor, input_length_tensor, output_tensor, mask,
                max_length
            ]

    # # 循环结束后，可能还有数据没有yield出去
    # if len(batch) != 0:
    #     # 下面的代码和上面的一致
    #     batch.sort(key=lambda x: len(x[0].split()), reverse=True)
    #     input_dialog_batch = []
    #     output_dialog_batch = []
    #     for pair in batch:
    #         input_dialog_batch.append(pair[0])
    #         output_dialog_batch.append(pair[1])
    #
    #     input_tensor, input_length_tensor = batchInput2paddedTensor(input_dialog_batch, voc)
    #     output_tensor, mask, max_length = batchOutput2paddedTensor(output_dialog_batch, voc)
    #
    #     yield [
    #         input_tensor, input_length_tensor, output_tensor, mask, max_length
    #     ]


# 根据encoder和decoder预测的网络
class GreedySearchDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 整个模型的前向逻辑
    def forward(self, input_seq, input_length, output_length, sos_token=1):
        """
        :param input_seq:  index序列 [1, max_length]
        :param input_length:  长度
        :param output_length: chatbot回复的句长
        :param sos_token: SOS占位符的index
        :return:
        """
        # 先在encoder中前向传播
        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)
        # 根据decoder的cell层数截取encoder_hidden作为decoder起始hidden
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # decoder最开始的input是一个SOS
        decoder_input = torch.tensor([[sos_token]])
        # 用来存储decoder得到的每个单词和这个单词的置信度
        predict_word_index = []
        predict_confidence = []
        for _ in range(output_length):
            # 经过decoder的前向计算
            decoder_output, current_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_output)
            max_possibility, index = torch.max(decoder_output, dim=1)
            predict_word_index.append(index.item())
            predict_confidence.append(max_possibility.item())
            # 传递下一次迭代的初值
            decoder_input = index.unsqueeze(0)
            decoder_hidden = current_hidden
        return predict_word_index, predict_confidence


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='一个batch中的对话数量（样本数量）')
    parser.add_argument('--save_dir', type=str, default='chatbot')
    parser.add_argument('--state_dict', type=str, default=None)

    parser.add_argument('--min_count', type=int, default=3, help='trim方法的修剪阈值')
    parser.add_argument('--max_length',
                        type=int,
                        default=20,
                        help='一个对话中每句话的最大句长')
    parser.add_argument('--teacher_forcing_ratio',
                        type=float,
                        default=0.9,
                        help='实行teacher_force_ratio的概率')

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=512,
                        help='RNN隐层维度，同时也是embedding的维度')
    parser.add_argument('--encoder_n_layers',
                        type=int,
                        default=2,
                        help='encoder的cell层数')
    parser.add_argument('--decoder_n_layers',
                        type=int,
                        default=2,
                        help='decoder的cell层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃参数的概率')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.003,
                        help='学习率')
    parser.add_argument('--clip', type=float, default=50.0, help='梯度裁剪阈值')

    parser.add_argument('--epoch', type=int, default=20, help='迭代轮数')
    parser.add_argument('--print_interval',
                        type=int,
                        default=900,
                        help='保存模型间隔')
    parser.add_argument('--save_interval', type=int, default=50, help='打印间隔')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--show_confidence', action='store_true')

    args = parser.parse_args()

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.test:
        state_dict_path = args.state_dict
        assert state_dict_path is not None, '--state_dict must be specify when in test mode!'

    if args.device == 'cuda':
        assert torch.cuda.is_available()

    return args


if __name__ == "__main__":
    BATCH_SIZE = 64  # 一个batch中的对话数量（样本数量）
    MAX_LENGTH = 20  # 一个对话中每句话的最大句长
    MIN_COUNT = 3  # trim方法的修剪阈值

    # 实例化词表
    voc = vocab(name="corpus",
                pad_token=PAD_token,
                sos_token=SOS_token,
                eos_token=EOS_token,
                unk_token=UNK_token)
    # 为词表载入数据，统计词频，并得到对话数据
    pairs = voc.load_data(path="./data/dialog.tsv")
    print("total number of dialogs:", len(pairs))

    # 修剪与替换
    pairs = trimAndReplace(voc, pairs, MIN_COUNT)

    # 获取loader
    loader = DataLoader(pairs, voc, batch_size=5)

    batch_item_names = [
        "input_tensor", "input_length_tensor", "output_tensor", "mask",
        "max_length"
    ]
    for batch_index, batch in enumerate(loader):
        for name, item in zip(batch_item_names, batch):
            print(f"\n{name} : {item}")
        break
