import torch
import torch.nn as nn

from neural_network import EncoderRNN, LuongAttentionDecoderRNN

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
            decoder_output, current_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            max_possibility, index = torch.max(decoder_output, dim=1)
            predict_word_index.append(index.item())
            predict_confidence.append(max_possibility.item())
            # 传递下一次迭代的初值
            decoder_input = index.unsqueeze(0)
            decoder_hidden = current_hidden
        return predict_word_index, predict_confidence


# 载入模型和字典
load_path = "./data/checkpoints/4500_checkpoint.tar"
checkpoint = torch.load(load_path, map_location=torch.device("cpu"))

encoder_state_dict = checkpoint["encoder"]
decoder_state_dict = checkpoint["decoder"]
embedding_state_dict = checkpoint["embedding"]
voc_dict = checkpoint["voc_dict"]

# 网络参数
hidden_size = 512                              # RNN隐层维度
encoder_n_layers = 2                         # encoder的cell层数
decoder_n_layers = 2                         # decoder的cell层数
dropout = 0.1                                      # 丢弃参数的概率

# 初始化，词向量矩阵、encoder、decoder并载入参数
embedding = torch.nn.Embedding(num_embeddings=voc_dict["num_words"],
                                                          embedding_dim=hidden_size,
                                                          padding_idx=voc_dict["pad_token"])
embedding.load_state_dict(embedding_state_dict)

encoder = EncoderRNN(hidden_size=hidden_size,
                                          embedding=embedding,
                                          n_layer=encoder_n_layers,
                                          dropout=dropout)

decoder = LuongAttentionDecoderRNN(score_name="concat",
                                                                    embedding=embedding,
                                                                    hidden_size=hidden_size,
                                                                    output_size=voc_dict["num_words"],
                                                                    n_layers=decoder_n_layers,
                                                                    dropout=dropout)

encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

encoder.eval()
decoder.eval()

# 实例化最终评估模型的实例
chatbot = GreedySearchDecoder(encoder, decoder)
MAX_LENGTH = 20             # 回复的最大句长

# 进入主循环
while True:
    user_dialog = input("User:").lower()
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

    print(f"chatbot: {' '.join(chatbot_words)}")

    # 将除去pad_token和eos_token的占位符后剩下的字符对应的置信度，保留3位小数输出
    print("chatbot's confidence: {}".format(
        list(map(lambda x : str(round(x * 100, 1)) + "%" , predict_confidence[:len(chatbot_words)]))
    ))

