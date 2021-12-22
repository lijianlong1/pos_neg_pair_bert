# Coding by long
# Datatime:2021/12/19 10:50
# Filename:dataloader.py
# Toolby: PyCharm
# ______________coding_____________
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def data_concat(data_less, data_more, bertfilepath,max_len):
    tokenizer = BertTokenizer.from_pretrained(bertfilepath)
    data_all_x_list = []
    data_all_y_list = []
    for i,j in zip(data_less, data_more):
        # 差一个数据清洗函数
        # 直接制作相关的数据和相应的编号
        data_less_token = tokenizer.encode(i, add_special_tokens=True) # 在句首和句尾加上特殊标志
        # 此处需要将所有的数据制作为定长
        while len(data_less_token) < max_len:
            data_less_token.append(0)  # 0代表padding操作
        data_less_token = data_less_token[:max_len]
        data_more_token = tokenizer.encode(j, add_special_tokens=True)
        while len(data_more_token) < max_len:
            data_more_token.append(0)  # 0 is padding
        data_more_token = data_more_token[:max_len]
        data_all_x_list.append(data_less_token)
        data_all_x_list.append(data_more_token)
        # 至此所有的数据都送到了data_all_x_list中
        data_all_y_list.append(0)
        data_all_y_list.append(1)
    print(len(data_all_x_list), len(data_all_y_list))
    return data_all_x_list, data_all_y_list


class Dataseter(Dataset):
    def __init__(self, filepath, bert_path, max_len = 512):
        # 制作相关的data时直接从使用pd.read_csv()读取相应的数据,还需要根据相应的BERT的token制作相应的词表映射，将文段转化为数字。
        data_frame_all = pd.read_csv(filepath)
        less_toxic = data_frame_all['less_toxic']
        more_toxic = data_frame_all['more_toxic']
        # 编号制作数据
        data_con_x, data_con_y = data_concat(less_toxic, more_toxic, bert_path, max_len)
        print(len(data_con_x), type(data_con_x), len(data_con_y), type(data_con_y))

        data_x_torch = torch.tensor(data_con_x).long()
        data_y_torch = torch.tensor(data_con_y).long()
        #print(data_x_torch.shape,data_y_torch.shape)
        self.len = len(data_con_x)
        self.data_x = data_x_torch  # 为一个torch格式的相关数组。
        self.data_y = data_y_torch  # 为一个torch数组

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len