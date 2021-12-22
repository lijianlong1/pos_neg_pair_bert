# Coding by long
# Datatime:2021/12/19 10:47
# Filename:trainer.py
# Toolby: PyCharm
# ______________coding_____________
import torch
import time
import numpy as np
import warnings
from traincode.models import BERT_Model_Classfy
from traincode.dataloader import Dataseter
from torch.utils.data import random_split,DataLoader
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup  # 引入相关的学习策略
from tqdm import tqdm  # 加入相应的进度条，用来显示训练的情况
warnings.filterwarnings('ignore')
# 使用相关的模块进行定义和训练
# 定义一些参数进行模型的训练
# 定义随机种子，保证代码可以复现
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
# 设置随机数种子


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
setup_seed(666)   # 设置随机数种子，确保实验结果可以复现
epochs = 5
batch_size = 1
lr = 2e-5
train_data_path = '../data/validation_data.csv'
bert_path = '../bertmodel/bert-base-uncased'
bert_vocab_path = '../bertmodel/bert-base-uncased/vocab.txt'
#制作相关的dataloader

data_all = Dataseter(train_data_path, bert_path=bert_vocab_path)  # 拿到所有的训练数据
# 按照8：2的概率划分为测试训练集和测试集
train_size = int(0.8*data_all.__len__())
dev_size = data_all.__len__() - train_size
# print(train_size,dev_size)  # 模型的输入的最大长度为不应该直接进行相应的划分，应该，制作成一对一对的数据进行训练和测试
train_data_set, dev_data_set = random_split(data_all, [train_size, dev_size])
train_dataloader = DataLoader(dataset=train_data_set, batch_size=batch_size, num_workers=2, shuffle=True)
dev_dataloader = DataLoader(dataset=dev_data_set, batch_size=batch_size, num_workers=2, shuffle=True)

# 至此我们的数据已经全部制作完成，需要进行模型的定义和损失函数与优化器的制作

model = BERT_Model_Classfy(bertmodelpath=bert_path)  # 至此模型已经成功加载进来了。
model.to(device)
# 损失函数和相关的优化器
criterion = torch.nn.BCEWithLogitsLoss()  # 使用单分类的bce-loss进行损失函数的
optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)  # weight_decay
# 在模型的训练过程中，增加相应的小trick
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader),
                                            num_training_steps=epochs * len(train_dataloader))

# 至此相关的损失函数和相应的优化器都
# 定义训练和模型的评估过程


def train(epochs=epochs):
    """
    # 定义训练函数，模型默认训练5轮，不改变相关的参数情况下
    # 后续
    :return: 模型的损失值，无法查看训练过程中的损失情况，每一轮的模型都保存一下。查看具体的训练效果
    """
    best_eval_loss = 100.0
    for epoch in range(epochs):  # 模型总共训练
        time_start = time.time()
        loss_sum = 0.0
        for index_id, data in enumerate(train_dataloader, 0):
            for i in tqdm(range(len(train_dataloader))):
                input_x, input_y = data  # 拿到相应的数据
                # 将数据送入device
                input_x, input_y = input_x.to(device), input_y.to(device)
                # print(input_x,input_y.shape)  # torch.Size([1]
                outputs = model(input_x)  # shape:batch-size * 1
                print(type(outputs))
                # print(outputs.shape)  # torch.Size([1, 1])
                loss_batch = criterion(outputs.float(), input_y.reshape(-1, 1).float())
                loss_batch.backward()  # 损失值反向传播
                optimizer.step()  # 梯度更新算法
                scheduler.step()  # 梯度更新策略
                # 至此模型训练的反向传播和梯度更新已经完成
                loss_sum += loss_batch.item()  # 将损失函数全部累加
                print(loss_batch.item())
        time_end = time.time()
        eval_loss = eval(model)
        if eval_loss < best_eval_loss:
            model_save_path = '../best_model_save/best_loss_model.pth'
            torch.save(model, model_save_path)
        print('这是第：%s次训练过程，训练总时长为%7.2f秒,训练的平均损失值为%f, 测试的损失值为：%f'
              % (str(epoch), (time_end-time_start), loss_sum/len(train_dataloader), eval_loss))


def eval(model):
    """
    # 相应的测试函数
    :return:
    """
    loss_sum = 0.0
    with torch.no_grad():
        for index_id, data_dev in enumerate(dev_dataloader, 0):  # 此时的数据为dev数据
            data_dev_x, data_dev_y = data_dev
            data_dev_x, data_dev_y = data_dev_x.to(device), data_dev_y.to(device)  # 将数据送入到device中
            output = model(data_dev_x)
            loss_batch = criterion(output.float(), data_dev_y.reshape(-1,1).float())
            loss_sum += loss_batch.item()  # 将所有的损失值都相加
        return loss_sum/len(dev_dataloader)  # 返回测试数据的平均损失值

if __name__=="__main__":
    train()
