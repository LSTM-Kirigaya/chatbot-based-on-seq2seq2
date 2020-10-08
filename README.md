# 基于Luong attention机制的seq2seq2的chatbot
使用seq2seq架构搭建聊天机器人。来源是pytorch的[chatbot_tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)

原版中使用了batch_first=False的数据组织方式，我则使用了batch_first=True的方式（单纯觉得batch_first=True更加符合我的理解）

encoder-decoder(Luong attention机制)的图大致如下：

![image.png](https://i.loli.net/2020/10/08/qaGL8uSsI9PTMir.png)
