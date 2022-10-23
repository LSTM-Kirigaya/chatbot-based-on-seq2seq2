# 基于Luong attention机制的seq2seq2的chatbot
使用seq2seq架构搭建聊天机器人。 灵感是pytorch的[chatbot_tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)

torch官方使用了`batch_first=False`的数据组织方式，我则使用了`batch_first=True`的方式（单纯觉得`batch_first=True`更加符合我的理解）

encoder-decoder(Luong attention机制)的图大致如下：

![image.png](https://i.loli.net/2020/10/08/qaGL8uSsI9PTMir.png)

使用Cornell的电影对话数据集，[链接](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) (项目中已经包含数据集，无需额外下载)

## 训练模型
```bash
python main.py --train --root="data" --device="cuda" --batch_size=64 --save_dir="chatbot" --teacher_forcing_ratio=0.95 --learning_rate=0.003 --epoch=15 --print_interval=100 --save_interval=1000
```

## 测试性能

```bash
python main.py --test --root="data" --state_dict=<saved state dict>
```

## 进行推理

推理为一问一答式的开放式问答

```bash
python play.py --state_dict=<saved state dict> --show_confidence
```

通过命令行来进行简单的交互：

![image.png](https://pic3.zhimg.com/80/v2-e129bc07bf6bc2c5c9ab468c410cb61b_720w.png)