# process.py
from itertools import zip_longest
import random
import torch

# 用来构造字典的类
class vocab(object):
    def __init__(self, name, pad_token, sos_token, eos_token, unk_token):
        self.name = name
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.trimmed = False                         # 代表这个词表对象是否经过了剪枝操作
        self.word2index = {"PAD" : pad_token, "SOS" : sos_token, "EOS" : eos_token, "UNK" : unk_token}
        self.word2count = {"UNK" : 0}
        self.index2word = {pad_token : "PAD", sos_token : "SOS", eos_token : "EOS", unk_token : "UNK"}
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
        if self.trimmed:   # 如果已经裁剪过了，那就直接返回
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
           keep_num, self.num_words - 4, keep_num / (self.num_words - 4)
        ))

        # 重构词表
        self.word2index  = {"PAD" : self.pad_token, "SOS" : self.sos_token, "EOS" : self.eos_token, "UNK" : self.unk_token}
        self.word2count = {}
        self.index2word = {self.pad_token : "PAD", self.sos_token : "SOS", self.eos_token : "EOS", self.unk_token : "UNK"}
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

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

# 将一句话转换成id序列(str->list)，结尾加上EOS
def sentenceToIndex(sentence, voc):
    return [voc.word2index[word] for word in sentence.split()] + [voc.eos_token]

# 将一个batch中的input_dialog转化为有pad填充的tensor，并返回tensor和记录长度的变量
# 返回的tensor是batch_first的
def batchInput2paddedTensor(batch, voc):
    # 先转换为id序列，但是这个id序列不对齐
    batch_index_seqs = [sentenceToIndex(sentence, voc) for sentence in batch]
    length_tensor = torch.tensor([len(index_seq) for index_seq in batch_index_seqs])
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
            batch.sort(key=lambda x : len(x[0].split()), reverse=True)
            input_dialog_batch = []
            output_dialog_batch = []
            for pair in batch:
                input_dialog_batch.append(pair[0])
                output_dialog_batch.append(pair[1])

            input_tensor, input_length_tensor = batchInput2paddedTensor(input_dialog_batch, voc)
            output_tensor, mask, max_length = batchOutput2paddedTensor(output_dialog_batch, voc)

            # 清空临时缓冲区
            batch = []

            yield [
                input_tensor, input_length_tensor, output_tensor, mask, max_length
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

if __name__ == "__main__":
    PAD_token = 0  # 补足句长的pad占位符的index
    SOS_token = 1  # 代表一句话开头的占位符的index
    EOS_token = 2  # 代表一句话结尾的占位符的index
    UNK_token = 3  # 代表不在词典中的字符
    BATCH_SIZE = 64  # 一个batch中的对话数量（样本数量）
    MAX_LENGTH = 20  # 一个对话中每句话的最大句长
    MIN_COUNT = 3  # trim方法的修剪阈值

    # 实例化词表
    voc = vocab(name="corpus", pad_token=PAD_token, sos_token=SOS_token, eos_token=EOS_token, unk_token=UNK_token)
    # 为词表载入数据，统计词频，并得到对话数据
    pairs = voc.load_data(path="./data/dialog.tsv")
    print("total number of dialogs:", len(pairs))

    # 修剪与替换
    pairs = trimAndReplace(voc, pairs, MIN_COUNT)

    # 获取loader
    loader = DataLoader(pairs, voc, batch_size=5)

    batch_item_names = ["input_tensor", "input_length_tensor", "output_tensor", "mask", "max_length"]
    for batch_index, batch in enumerate(loader):
        for name, item in zip(batch_item_names, batch):
            print(f"\n{name} : {item}")
        break
