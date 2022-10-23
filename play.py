import torch
from colorama import Style, Fore

from model import EncoderRNN, LuongAttentionDecoderRNN
from util import parse_arg, GreedySearchDecoder

if __name__ == "__main__":
    args = parse_arg()

    assert args.state_dict is not None
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

    encoder = EncoderRNN(hidden_size=hidden_size, embedding=embedding, n_layers=encoder_n_layers, dropout=dropout)

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

    caption = '|{}{}{}|'.format(' ' * 3, "chat begins, press 'q' to quit the conversation", ' ' * 3)
    print("-" * len(caption))
    print(caption)
    print("-" * len(caption))

    # 进入主循环
    while True:
        user_dialog = input(f" {Fore.YELLOW}User:{Style.RESET_ALL} ").lower()
        if user_dialog == "q" or user_dialog == "quit":
            break
        # word转index， 并将不在词表中的单词替换成unk_token
        input_seq = [voc_dict["word2index"].get(word, voc_dict["unk_token"]) for word in user_dialog.split()]
        input_length = torch.tensor([len(input_seq)])
        input_seq = torch.tensor(input_seq).unsqueeze(0)
        # 将index序列输入chatbot，获取回复index序列
        predict_indexes, predict_confidence = chatbot(input_seq, input_length, MAX_LENGTH)
        # 将chatbot回复的index序列转为word，并将代表pad_token和eos_token的index去除
        chatbot_words = []
        for index in predict_indexes:
            if index in [voc_dict["pad_token"], voc_dict["eos_token"]]:
                continue
            else:
                chatbot_words.append(voc_dict["index2word"][index])

        reply = ' '.join(chatbot_words).strip()
        print(Fore.BLUE, "chatbot:", Style.RESET_ALL, reply)

        # 将除去pad_token和eos_token的占位符后剩下的字符对应的置信度，保留3位小数输出
        if args.show_confidence:
            print(
                Fore.MAGENTA, "chatbot's confidence: {}".format(
                    list(map(lambda x: str(round(x * 100, 1)) + "%", predict_confidence[:len(chatbot_words)]))),
                Style.RESET_ALL)
