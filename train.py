# train.py
import torch
import random
from time import time
import os
import math

# 导入之前写好的函数或类
from process import vocab, trimAndReplace, DataLoader
from neural_network import EncoderRNN, LuongAttentionDecoderRNN

# 定义运算设备
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 常量
PAD_token = 0  # 补足句长的pad占位符的index
SOS_token = 1  # 代表一句话开头的占位符的index
EOS_token = 2  # 代表一句话结尾的占位符的index
UNK_token = 3  # 代表不在词典中的字符
# 超参
BATCH_SIZE = 64                               # 一个batch中的对话数量（样本数量）
MAX_LENGTH = 20                             # 一个对话中每句话的最大句长
MIN_COUNT = 3                                 # trim方法的修剪阈值
TEACHER_FORCING_RATIO = 1.0      # 实行teacher_force_ratio的概率
LEARNING_RATE = 0.0001               # 学习率
CLIP = 50.0                                           # 梯度裁剪阈值
# 网络参数
hidden_size = 512                              # RNN隐层维度
encoder_n_layers = 2                         # encoder的cell层数
decoder_n_layers = 2                         # decoder的cell层数
dropout = 0.1                                      # 丢弃参数的概率

# 轮数与打印间隔
epoch_num = 15                  # 迭代轮数
print_interval = 50              # 打印间隔
save_interval = 900           # 保存模型间隔

mask_loss_all = []


# 使用mask loss的方法避免计算pad的loss
# 计算非pad对应位置的交叉熵（负对数似然）
def maskNLLLoss(output, target, mask):
    """
    :param output: decoder的所有output的拼接    [batch_size, max_length, output_size]
    :param target: 标签，也就是batch的output_dialog对应的id序列  [batch_size, max_length]
    :param mask: mask矩阵    与target同形
    :return: 交叉熵损失、单词个数
    """

    target = target.type(torch.int64).to(device)
    mask = mask.type(torch.BoolTensor).to(device)

    total_word = mask.sum()  # 单词个数
    crossEntropy = -torch.log(torch.gather(output, dim=2, index=target.unsqueeze(2)))
    # crossEntropy : [batch_size, max_length, 1]
    loss = crossEntropy.squeeze(2).masked_select(mask).mean()
    loss = loss.to(device)
    return loss, total_word.item()

# 定义训练一个batch的逻辑
# 在这个batch中更新网络时，有一定的概率会使用teacher forcing的方法来加速收敛， 这个概率为teacher_forcing_ratio
def trainOneBatch(input_seq, input_length, target, mask, max_target_len,
                  encoder, decoder, encoder_optimizer, decoder_optimizer,
                  batch_size, clip, teacher_forcing_ratio,
                  encoder_lr_scheduler=None, decoder_lr_scheduler=None):
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

    decoder_output = torch.FloatTensor()    # 用来保存decoder的所有输出
    decoder_output = decoder_output.to(device)

    # encoder的前向计算
    encoder_output, encoder_hidden = encoder(input_seq, input_length)

    # 为decoder的计算初始化一个开头SOS
    decoder_input = torch.tensor([SOS_token for _ in range(batch_size)]).reshape([-1, 1])
    decoder_input = decoder_input.to(device)

    # 根据decoder的cell层数截取encoder的h_n的层数
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 根据概率决定是否使用teacher force
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for i in range(max_target_len):
            # 执行一步decoder前馈，只计算了一个单词
            output, current_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            # output : [batch_size, output_size]
            # current_hidden : [1, batch_size, hidden_size]
            # 将本次time_step得到的结果放入decoder_output中
            decoder_output = torch.cat([decoder_output, output.unsqueeze(1)], dim=1)
            # 使用teacher forcing：使用target（真实的单词）作为decoder的下一次的输入，而不是我们计算出的预测值（计算出的概率最大的单词）
            decoder_input = target[:, i].reshape([-1, 1])
            decoder_hidden = current_hidden
    else:
        for i in range(max_target_len):
            # 执行一步decoder前馈，只计算了一个单词
            output, current_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            # output : [batch_size, output_size]
            # current_hidden : [1, batch_size, hidden_size]
            # 从softmax的结果中得到预测的index
            predict_index = torch.argmax(output, dim=1)
            # 将本次time_step得到的结果放入decoder_output中
            decoder_output = torch.cat([decoder_output, output.unsqueeze(1)], dim=1)
            decoder_input = predict_index.reshape([-1, 1])
            decoder_hidden = current_hidden

    # 计算本次batch中的mask_loss和总共的单词数
    mask_loss, word_num = maskNLLLoss(output=decoder_output, target=target, mask=mask)

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

# 开始训练
print("build vocab_list...")
# 首先构建字典
voc = vocab(name="corpus", pad_token=PAD_token, sos_token=SOS_token, eos_token=EOS_token, unk_token=UNK_token)
# 载入数据
pairs = voc.load_data(path="./data/dialog.tsv")
print(f"load {len(pairs)} dialogs successfully")
# 对数据进行裁剪
pairs = trimAndReplace(voc=voc, pairs=pairs, min_count=MIN_COUNT)
print(f"there are {voc.num_words} words in the vocab")

# 定义词向量嵌入矩阵
embedding = torch.nn.Embedding(num_embeddings=voc.num_words,
                                                          embedding_dim=hidden_size,
                                                          padding_idx=PAD_token)
# 获取encoder和decoder
encoder = EncoderRNN(hidden_size=hidden_size,
                                          embedding=embedding,
                                          n_layer=encoder_n_layers,
                                          dropout=dropout)

decoder = LuongAttentionDecoderRNN(score_name="concat",
                                                                    embedding=embedding,
                                                                    hidden_size=hidden_size,
                                                                    output_size=voc.num_words,
                                                                    n_layers=decoder_n_layers,
                                                                    dropout=dropout)
encoder = encoder.to(device)
decoder = decoder.to(device)

# 为encoder和decoder分别定义优化器
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

# Leslie的Triangle2学习率衰减方法
def Triangular2(T_max, gamma):
    def new_lr(step):
        region = step // T_max + 1    # 所处分段
        increase_rate = 1 / T_max * math.pow(gamma, region - 2)    # 增长率的绝对值
        return increase_rate * (step - (region - 1) * T_max) if step <= (region - 0.5) * T_max else - increase_rate * (step - region * T_max)
    return new_lr


# 定义优化器学习率的衰减策略
encoder_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=encoder_optimizer,
                                                                                                   lr_lambda=Triangular2(T_max=300, gamma=0.5))
decoder_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=decoder_optimizer,
                                                                                                   lr_lambda=Triangular2(T_max=300, gamma=0.5))


global_step = 0
start_time = time()
encoder.train()
decoder.train()

print("start to train...")

# 轮数遍历
for epoch in range(epoch_num):
    print("-" * 20)
    print("Epoch : ",epoch)
    # 获取loader
    train_loader = DataLoader(pairs=pairs, voc=voc, batch_size=BATCH_SIZE, shuffle=True)
    # 遍历生成器
    for batch_num, batch in enumerate(train_loader):
        global_step += 1
        # batch中的信息 : ["input_tensor", "input_length_tensor", "output_tensor", "mask", "max_length"]
        loss = trainOneBatch(input_seq=batch[0], input_length=batch[1], target=batch[2], mask=batch[3], max_target_len=batch[4],
                                          encoder=encoder, decoder=decoder,  encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
                                          batch_size=BATCH_SIZE, clip=CLIP, teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                                          encoder_lr_scheduler=encoder_lr_scheduler, decoder_lr_scheduler=decoder_lr_scheduler)

        mask_loss_all.append(loss)

        if global_step % print_interval == 0:
            print("Epoch : {}\tbatch_num : {}\tloss: {:.6f}\ttime point : {:.2f}s\tmodel_lr : {:.10f}".format(
                epoch, batch_num, loss, time() - start_time, encoder_optimizer.param_groups[0]["lr"]
            ))

        # 将check_point存入./data/check_points这个文件夹中
        if global_step % save_interval == 0:

            # 先判断目标路径是否存在，不存在则创建
            if not os.path.exists("./data/checkpoints"):
                os.makedirs("./data/checkpoints")

            check_point_save_path = f"./data/checkpoints/{global_step}_checkpoint.tar"

            torch.save({
                "iteration" : global_step,
                "encoder" : encoder.state_dict(),
                "decoder" : decoder.state_dict(),
                "encoder_optimizer" : encoder_optimizer.state_dict(),
                "decoder_optimizer" : decoder_optimizer.state_dict(),
                "encoder_lr_scheduler" : encoder_lr_scheduler.state_dict(),
                "decoder_lr_scheduler" : decoder_lr_scheduler.state_dict(),
                "loss" : loss,
                "voc_dict" : voc.__dict__,
                "embedding" : embedding.state_dict()
            }, check_point_save_path)

            print(f"save model to {check_point_save_path}")