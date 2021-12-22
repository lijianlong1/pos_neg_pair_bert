# Coding by long
# Datatime:2021/12/19 10:50
# Filename:models.py
# Toolby: PyCharm
# ______________coding_____________
import torch.nn as nn
from transformers import BertModel


class BERT_Model_Classfy(nn.Module):
    def __init__(self, bertmodelpath, numclass=1):
        super().__init__()
        self.BERT_model = BertModel.from_pretrained(bertmodelpath)
        # 加载相应的预训练模型
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.fc = nn.Linear(768, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(64, numclass)

    def forward(self, x):  # mask在此不传入，直接将相关的模型的，此处传入到BERT模型的中的数据为数字序列
        x = self.BERT_model(x)  # 这是最后一层相关数据
        x = x.last_hidden_state
        print(x.shape)
        x = x[:, 0, :]  # 拿到cls对应的参数
        print(x.shape)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.fc1(x)  # 最后拿到相应的模型进行训练处理。
        return x  # 此时的x为一个

# 至此bert模型已经得到，且直接使用CLS对相关的模块进行分类。


