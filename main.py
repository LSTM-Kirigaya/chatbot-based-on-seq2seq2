from time import time
import os

import torch
from nltk.translate.bleu_score import sentence_bleu

# 导入之前写好的函数或类
from util import (vocab, GreedySearchDecoder, PAD_token, SOS_token, EOS_token,
                  UNK_token, parse_arg, trimAndReplace, DataLoader,
                  train_one_batch)
from model import EncoderRNN, LuongAttentionDecoderRNN


def train(args):
    # 超参
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    MIN_COUNT = args.min_count
    TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
    LEARNING_RATE = args.learning_rate
    CLIP = args.clip

    # 网络参数
    hidden_size = args.hidden_dim
    encoder_n_layers = args.encoder_n_layers
    decoder_n_layers = args.decoder_n_layers
    dropout = args.dropout

    # 轮数与打印间隔
    epoch = args.epoch
    print_interval = args.print_interval
    save_interval = args.save_interval

    # 设备
    device = args.device
    save_dir = args.save_dir
    root = args.root

    mask_loss_all = []

    # 定义训练一个batch的逻辑
    # 在这个batch中更新网络时，有一定的概率会使用teacher forcing的方法来加速收敛， 这个概率为teacher_forcing_ratio

    # 开始训练
    print("build vocab_list...")
    # 首先构建字典
    voc = vocab(name="corpus",
                pad_token=PAD_token,
                sos_token=SOS_token,
                eos_token=EOS_token,
                unk_token=UNK_token)
    # 载入数据
    data_path = os.path.join(root, 'dialog.tsv')
    pairs = voc.load_data(data_path)

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
                         n_layers=encoder_n_layers,
                         dropout=dropout)

    decoder = LuongAttentionDecoderRNN(score_name="dot",
                                       embedding=embedding,
                                       hidden_size=hidden_size,
                                       output_size=voc.num_words,
                                       n_layers=decoder_n_layers,
                                       dropout=dropout)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 为encoder和decoder分别定义优化器
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                                         lr=LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(),
                                         lr=LEARNING_RATE)

    # 定义优化器学习率的衰减策略
    # encoder_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=encoder_optimizer,
    # #                                                                                                    lr_lambda=Triangular2(T_max=300, gamma=0.5))
    # # decoder_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=decoder_optimizer,
    # #                                                                                                    lr_lambda=Triangular2(T_max=300, gamma=0.5))

    encoder_lr_scheduler = None
    decoder_lr_scheduler = None

    global_step = 0
    start_time = time()
    encoder.train()
    decoder.train()

    print("start to train...")

    # 轮数遍历
    for epoch in range(epoch):
        print("-" * 20)
        print("Epoch : ", epoch)
        # 获取loader
        train_loader = DataLoader(pairs=pairs,
                                  voc=voc,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
        # 遍历生成器
        for batch_num, batch in enumerate(train_loader):
            global_step += 1
            # batch中的信息 : ["input_tensor", "input_length_tensor", "output_tensor", "mask", "max_length"]
            loss = train_one_batch(input_seq=batch[0],
                                   input_length=batch[1],
                                   device=device,
                                   target=batch[2],
                                   mask=batch[3],
                                   max_target_len=batch[4],
                                   encoder=encoder,
                                   decoder=decoder,
                                   encoder_optimizer=encoder_optimizer,
                                   decoder_optimizer=decoder_optimizer,
                                   batch_size=BATCH_SIZE,
                                   clip=CLIP,
                                   teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                                   encoder_lr_scheduler=encoder_lr_scheduler,
                                   decoder_lr_scheduler=decoder_lr_scheduler)

            mask_loss_all.append(loss)

            if global_step % print_interval == 0:
                print(
                    "Epoch : {}\tbatch_num : {}\tloss: {:.6f}\ttime point : {:.2f}s\tmodel_lr : {:.10f}"
                    .format(epoch, batch_num, loss,
                            time() - start_time,
                            encoder_optimizer.param_groups[0]["lr"]))

            # 将check_point存入./data/check_points这个文件夹中
            if global_step % save_interval == 0:
                checkpoint_name = f'{global_step}_checkpoint.tar'
                check_point_save_path = os.path.join(save_dir, checkpoint_name)

                torch.save(
                    {
                        "iteration": global_step,
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "encoder_optimizer": encoder_optimizer.state_dict(),
                        "decoder_optimizer": decoder_optimizer.state_dict(),
                        # "encoder_lr_scheduler" : encoder_lr_scheduler.state_dict(),
                        # "decoder_lr_scheduler" : decoder_lr_scheduler.state_dict(),
                        "loss": loss,
                        "voc_dict": voc.__dict__,
                        "embedding": embedding.state_dict()
                    },
                    check_point_save_path)

                print(f"save model to {check_point_save_path}")


def test(args):
    # 载入模型和字典
    load_path = args.state_dict
    MAX_LENGTH = args.max_length

    # 网络参数
    hidden_size = args.hidden_dim
    encoder_n_layers = args.encoder_n_layers
    decoder_n_layers = args.decoder_n_layers
    dropout = args.dropout

    # 模型有可能是在gpu上训练的，需要先把模型参数转换成cpu可以运算的类型
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))

    encoder_state_dict = checkpoint["encoder"]
    decoder_state_dict = checkpoint["decoder"]
    embedding_state_dict = checkpoint["embedding"]
    voc_dict = checkpoint["voc_dict"]

    # 初始化，词向量矩阵、encoder、decoder并载入参数
    embedding = torch.nn.Embedding(num_embeddings=voc_dict["num_words"],
                                   embedding_dim=hidden_size,
                                   padding_idx=voc_dict["pad_token"])
    embedding.load_state_dict(embedding_state_dict)

    encoder = EncoderRNN(hidden_size=hidden_size,
                         embedding=embedding,
                         n_layers=encoder_n_layers,
                         dropout=dropout)

    decoder = LuongAttentionDecoderRNN(score_name="dot",
                                       embedding=embedding,
                                       hidden_size=hidden_size,
                                       output_size=voc_dict["num_words"],
                                       n_layers=decoder_n_layers,
                                       dropout=dropout)

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    # 设为评估模式，网络参数停止更新
    encoder.eval()
    decoder.eval()

    # 实例化最终评估模型的实例
    chatbot = GreedySearchDecoder(encoder, decoder)

    data_path = os.path.join(args.root, 'dialog.tsv')
    bleus = []

    for line in open(data_path, 'r', encoding='utf-8'):
        try:
            input_dialog, output_dialog = line.strip().split("\t")
            reference = [output_dialog.split()]
            input_seq = [
                voc_dict["word2index"].get(word, voc_dict["unk_token"])
                for word in input_dialog.split()
            ]
            input_length = torch.tensor([len(input_seq)])
            input_seq = torch.tensor(input_seq).unsqueeze(0)
            predict_indexes, _ = chatbot(input_seq, input_length, MAX_LENGTH)
            # 将chatbot回复的index序列转为word，并将代表pad_token和eos_token的index去除
            candidate = []
            for index in predict_indexes:
                if index in [voc_dict["pad_token"], voc_dict["eos_token"]]:
                    continue
                else:
                    candidate.append(voc_dict["index2word"][index])

            bleu = sentence_bleu(reference, candidate)
            bleus.append(bleu)

        except:
            pass

    print('avg bleu', sum(bleus) / len(bleus))


if __name__ == "__main__":
    torch.manual_seed(1024)

    # get args
    args = parse_arg()
    for arg_name, value in args._get_kwargs():
        if isinstance(value, bool) and value:
            globals()[arg_name](args)
