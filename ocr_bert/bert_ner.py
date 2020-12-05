#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

#BERT的相关参数
mode = 0
maxlen = 300
learning_rate = 5e-5
min_learning_rate = 1e-6

dir_path = os.getcwd()
print(dir_path)

config_path = dir_path+'/best_bert_model/chinese/bert_config.json'
checkpoint_path = dir_path+'/best_bert_model/chinese/bert_model.ckpt'
dict_path = dir_path+'/best_bert_model/chinese/vocab.txt'


# In[2]:


token_dict = {}

# 加载词表
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    # 定制化分词器，这里不论中文还是英文都根据单个字符进行切分
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

# 构造分词器实例
tokenizer = OurTokenizer(token_dict)

def seq_padding(X, padding=0):
    # 填充补0
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])    

def list_find(list1, list2):
    # 在list1中查找子串list2，如果找到返回初始的下标，否则返回-1
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1


# 获取训练集
#训练集字段介绍
#id代表唯一数据标识
#title和text是用于识别的文本，可能为空
#unknownEntities代表实体，可能有多个，通过英文";"分隔
train_data = pd.read_csv(dir_path+'/best_bert_model/data/weiboData/Train_Data1.csv').fillna('>>>>>')
train_data = train_data[~train_data['unknownEntities'].isnull()].reset_index(drop = True)

# 将title和text合并成content字段，将模型转化成单输入问题
# 如果title和text字段相等那么就合并，否则返回其中一个就行了
train_data['content'] = train_data.apply(lambda x: x['title'] if x['title']==x['text'] else x['title']+x['text'], axis = 1)

# 对于unknownEntities字段中存在多个实体的只使用第一个实体
train_data['unknownEntity'] = train_data['unknownEntities'].apply(lambda x:x.split(';')[0])

# 获取所有的实体类别
# 这里先将unknownEntities进行拼接，然后根据";"切分
entity_str = ''
for i in train_data['unknownEntities'].unique():
    entity_str = i + ';' + entity_str  
    
entity_classes_full = set(entity_str[:-1].split(";"))
# 3183

# 训练集变成了两个字段：
# 需要识别的文本content，这是原始数据集中title和text合并之后的数据
# 未知实体列表unknownEntities，类似于label，只会有一个实体
train_data_list = []
for content,entity in zip(train_data['content'], train_data['unknownEntity']):
    train_data_list.append((content, entity))


# 找到训练集中content字段中文、英文和数字以外的特殊字符
additional_chars = set()
for data in train_data_list:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', data[1]))
# print(additional_chars)

# 构建训练模型
# 整个模型是单输入和单输出的问题
# 模型输入是一条query文本，这里会先将文本转换成三层embedding，token embedding、seg embedding和position embedding
# 因为句子关系可以直接获取，所以只返回token embedding、seg embedding两个输入，作为网络的输入
# 模型输出是一个实体，这个实体是query中的一个子片段
#根据这个输出特性，输出应该用指针结构，通过两个Softmax分别预测首尾，然后得到一个实体
# 所以这里返回实体的左边界和右边界作为网络的输出

# 导入预训练模型
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

# # 是否进行微调
for layer in bert_model.layers:
    layer.trainable = True

# 词编码输入
word_in = Input(shape=(None,), name='word_in')
# 句子对编码输入
seg_in = Input(shape=(None,), name='seg_in')
# 实体左边界数组，只有实体开始位置为1，其他均为0
entiry_left_in = Input(shape=(None,), name='entiry_left_in')
# 实体右边界数组，只有实体结束位置为1，其他均为0
entiry_right_in = Input(shape=(None,), name='entiry_right_in')

x1, x2, s1, s2 = word_in, seg_in, entiry_left_in, entiry_right_in

bert_in = bert_model([word_in, seg_in])
ps1 = Dense(1, use_bias=False, name='ps1')(bert_in)
# 遮掩掉不应该读取到的信息，或者无用的信息，以0作为mask的标记
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x_mask')(word_in)
ps2 = Dense(1, use_bias=False, name='ps2')(bert_in)
ps11 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='ps11')([ps1, x_mask])
ps22 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='ps22')([ps2, x_mask])

train_model = Model([word_in, seg_in, entiry_left_in, entiry_right_in], [ps11, ps22])

# 构建模型
build_model = Model([word_in, seg_in], [ps11, ps22])

loss1 = K.mean(K.categorical_crossentropy(entiry_left_in, ps11, from_logits=True))
ps22 -= (1 - K.cumsum(s1, 1)) * 1e10
loss2 = K.mean(K.categorical_crossentropy(entiry_right_in, ps22, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


# In[9]:


# 经过一个softmax操作
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)
softmax([1, 9, 5, 3])


# 抽取实体测试
# 输入文本
# 返回实体列表，这里最多返回num个实体
def extract_entity_test(model, text_in, num):
    text_in = text_in[:maxlen]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2  = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    
    # 特殊字符转换为负值
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
            
    tg_list = list()
    
    for i in range(num):
        #[0.99977237, 0.00011352481, 4.0782343e-05, 2.4224111e-05, 1.7350189e-05, 1.0297682e-05, 8.015117e-06, 6.223183e-06
        #, 3.117688e-06, 1.7270181e-06, 1.125549e-06, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]
        #_ps1中的值代表为实体的概率得分，越大越可能是实体的左边界
        #将_ps1按概率值值降序排序
        #num代表选择topN个实体
        start = np.argwhere((_ps1==sorted(_ps1,reverse=True)[i]))[0][0]
        
        # 设置中断的条件，当字符的长度为1并且为特殊字符并且不属于正常字符
        for end in range(start, len(_tokens)):
            _t = _tokens[end]
            if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
                break
        # _ps2中的值代表为实体的概率得分
        # argmax()是返回_ps2最大值的索引        
        end = _ps2[start:end+1].argmax() + start
        a = text_in[start-1: end]
        tg_list.append(a)
        tg_list = list(set(tg_list))
        print(i, start, end,a )
    return ';'.join(tg_list)



# 导入模型权重
build_model.load_weights(dir_path+'/best_bert_model/model/best_model_12.weights')

# import time
#
# start_time = time.time()
# extract_entity_test(build_model, '主营业:宵夜.烧烤.蛋炒饭,有家烧烤店， ', 1)
# end_time = time.time()
#
# print("time %d" % (end_time-start_time))
def bert_ner_text(str_list):
    company_text = extract_entity_test(build_model, str_list, 1)
    return company_text


