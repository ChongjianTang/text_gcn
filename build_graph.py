import os
import random
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

import sys


def create_test_index_file(dataset, test_ids):
    test_ids_str = '\n'.join(str(index) for index in test_ids)  # 把测试集数据对应的索引转成字符串 并用\n连接在一起
    # 把测试集数据对应的id写入文件
    with open('./data/' + dataset + '.test.index', 'w') as f:
        f.write(test_ids_str)


def create_train_index_file(dataset, train_ids):
    train_ids_str = '\n'.join(str(index) for index in train_ids)  # 把训练集数据对应的索引转成字符串 并用\n连接在一起
    # 把训练集数据对应的id写入文件
    with open('./data/' + dataset + '.train.index', 'w') as f:
        f.write(train_ids_str)


def create_shuffled_file(dataset, shuffle_doc_tag_list, shuffle_doc_words_list):
    # 用\n连接
    shuffle_doc_tag_str = '\n'.join(shuffle_doc_tag_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    # 写入打乱后的数据集 对应的分割信息和类别信息
    with open('./data/' + dataset + '_shuffled.txt', 'w') as f:
        f.write(shuffle_doc_tag_str)
    # 写入打乱的处理好的数据集
    with open('./data/corpus/' + dataset + '_shuffled.txt', 'w') as f:
        f.write(shuffle_doc_words_str)


def create_vocab_file(dataset, vocab):
    vocab_str = '\n'.join(vocab)  # 用\n连接词典中的词
    # 把数据集对应的字典写入文件
    with open('./data/corpus/' + dataset + '_vocab.txt', 'w') as f:
        f.write(vocab_str)


def create_labels_file(dataset, label_list):
    label_list_str = '\n'.join(label_list)  # 用\n连接 一行一个
    # 把数据集对应的标签列表(去重)写入文件
    with open('./data/corpus/' + dataset + '_labels.txt', 'w') as f:
        f.write(label_list_str)


def create_real_train_tag_file(dataset, real_train_doc_tags):
    real_train_doc_names_str = '\n'.join(real_train_doc_tags)
    # 写入真实训练集对应的信息
    with open('./data/' + dataset + '.real_train.name', 'w') as f:
        f.write(real_train_doc_names_str)


if __name__ == '__main__':
    from utils.utils import loadWord2Vec, clean_str

    if len(sys.argv) != 2:  # 执行脚本需要指定数据集 python后面需要有两个参数  一个是.py脚本 另一个是要处理的数据集名称
        sys.exit("Use: python build_graph.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']  # 全部5个数据集
    # build corpus
    dataset = sys.argv[1]  # 获取待处理的数据集名称

    if dataset not in datasets:  # 数据集名称设置错误
        sys.exit("wrong dataset name")

    # Read Word Vectors
    # word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
    # word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    # _, embd, word_vector_map = loadWord2Vec(word_vector_file)
    # word_embeddings_dim = len(embd[0])

    word_embeddings_dim = 300  # 词嵌入的维度
    word_vector_map = {}  # 词到词嵌入/向量的映射字典

    # shuffling
    doc_tag_list = []
    doc_train_list = []
    doc_test_list = []

    train_ids = []
    test_ids = []

    # 读取数据集的分割信息(训练集 or 测试集)和类别信息
    with open('./data/' + dataset + '.txt', 'r') as f:
        lines = f.readlines()  # 把所有行读到一个列表中
        for line in lines:
            # 按行处理
            doc_tag_list.append(line.strip())
            temp = line.split("\t")
            if temp[1] == 'test':
                doc_test_list.append(line.strip())  # 测试集对应的数据
                test_ids.append(temp[0])
            elif temp[1] == 'train':
                doc_train_list.append(line.strip())  # 训练集对应的数据
                train_ids.append(temp[0])

    random.shuffle(train_ids)  # 随机打乱
    random.shuffle(test_ids)  # 随机打乱
    # print(doc_train_list)
    # print(doc_test_list)

    doc_content_list = []
    # 读取之前处理好的数据集
    with open('./data/corpus/' + dataset + '.clean.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
    # print(doc_content_list)

    # train_ids = []  # 存储训练集数据对应的id
    # for train_name in doc_train_list:
    #     train_id = doc_name_list.index(train_name)
    #     train_ids.append(train_id)
    # print(train_ids)
    # random.shuffle(train_ids)  # 随机打乱

    # partial labeled data
    # train_ids = train_ids[:int(0.2 * len(train_ids))]

    # test_ids = []  # 存储训练集数据对应的id
    # for test_name in doc_test_list:
    #     test_id = doc_name_list.index(test_name)
    #     test_ids.append(test_id)
    # print(test_ids)
    # random.shuffle(test_ids)  # 随机打乱

    create_train_index_file(dataset, train_ids)
    create_test_index_file(dataset, test_ids)

    ids = train_ids + test_ids
    # print(ids)
    print(len(ids))

    shuffle_doc_tag_list = []
    shuffle_doc_words_list = []
    # 打乱处理好的数据集以及数据集的分割信息和类别信息
    for id in ids:
        shuffle_doc_tag_list.append(doc_tag_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    create_shuffled_file(dataset, shuffle_doc_tag_list, shuffle_doc_words_list)

    # build vocab 建立词典
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        # 统计词频
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)  # 去重后的词典
    vocab_size = len(vocab)  # 词典大小

    word_doc_list = {}

    # 统计每个单词在哪些文档中出现过
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word not in appeared:
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

    word_doc_freq = {}  # 统计每个单词 出现在了几个文档中
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}  # 词到索引的映射
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    create_vocab_file(dataset, vocab)

    '''
    Word definitions begin
    '''
    '''
    definitions = []
    
    for word in vocab:
        word = word.strip()
        synsets = wn.synsets(clean_str(word))
        word_defs = []
        for synset in synsets:
            syn_def = synset.definition()
            word_defs.append(syn_def)
        word_des = ' '.join(word_defs)
        if word_des == '':
            word_des = '<PAD>'
        definitions.append(word_des)
    
    string = '\n'.join(definitions)
    
    
    f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
    f.write(string)
    f.close()
    
    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vec.fit_transform(definitions)
    tfidf_matrix_array = tfidf_matrix.toarray()
    print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))
    
    word_vectors = []
    
    for i in range(len(vocab)):
        word = vocab[i]
        vector = tfidf_matrix_array[i]
        str_vector = []
        for j in range(len(vector)):
            str_vector.append(str(vector[j]))
        temp = ' '.join(str_vector)
        word_vector = word + ' ' + temp
        word_vectors.append(word_vector)
    
    string = '\n'.join(word_vectors)
    
    f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
    f.write(string)
    f.close()
    
    word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    _, embd, word_vector_map = loadWord2Vec(word_vector_file)
    word_embeddings_dim = len(embd[0])
    '''

    '''
    Word definitions end
    '''

    # label list
    # 得到每条数据/文档对应的标签列表 （去重）
    label_set = set()
    for doc_meta in shuffle_doc_tag_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    create_labels_file(dataset, label_list)

    # x: feature vectors of training docs, no initial features
    # select 90% training set
    train_size = len(train_ids)  # 训练集大小
    val_size = int(0.1 * train_size)  # 验证集大小 10%
    real_train_size = train_size - val_size  # - int(0.1 * train_size) 真实训练集大小
    # different training rates

    real_train_doc_tags = shuffle_doc_tag_list[:real_train_size]

    create_real_train_tag_file(dataset, real_train_doc_tags)

    row_x = []
    col_x = []
    data_x = []
    # 把真实训练集中每条数据/文档表示成一个向量
    # 整个数据集就是一个矩阵 存储为稀疏矩阵格式
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # 文档向量初始化为0
        doc_words = shuffle_doc_words_list[i]  # 遍历每个文档/数据
        words = doc_words.split()  # 分词
        doc_len = len(words)  # 文档包含的词数
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]  # 拿到每个词对应的词向量
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)  # 文档向量等于 文档中每个词的词向量之和

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)

    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, word_embeddings_dim))

    y = []
    # 把真实训练集 每条数据/文档对应的标签 转换为one-hot
    for i in range(real_train_size):
        doc_meta = shuffle_doc_tag_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    print(y)

    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    # 把测试集中每条数据/文档表示成一个向量
    # 整个测试集就是一个矩阵 存储为稀疏矩阵格式
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    # 把测试集 每条数据/文档对应的标签 转换为one-hot
    for i in range(test_size):
        doc_meta = shuffle_doc_tag_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)
    print(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words

    # 随机初始化词嵌入矩阵
    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))

    # 用预训练词向量 对 词嵌入矩阵进行覆盖
    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    row_allx = []
    col_allx = []
    data_allx = []
    # 把全部训练集中每条数据/文档表示成一个向量
    # 整个数据集就是一个矩阵 存储为稀疏矩阵格式
    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

    # 在全部训练集对应的矩阵后再拼接上 词嵌入矩阵
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    # 把拼接后的矩阵转成稀疏格式
    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    # 把全部训练集 每条数据/文档对应的标签 转换为one-hot
    for i in range(train_size):
        doc_meta = shuffle_doc_tag_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    # 对词典中的每个词 也添加一个对应的类别one-hot向量 全部初始化为0
    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    # x (real_train_size, word_embeddings_dim)
    # y (real_train_size, num_classes)
    # tx (test_size, word_embeddings_dim)
    # ty (test_size, num_classes)
    # allx (train_size + vocab_size, word_embeddings_dim)
    # ally (train_size + vocab_size, num_classes)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)




    '''
    Doc word heterogeneous graph 文档-词异构图
    '''

    # word co-occurrence with context windows
    window_size = 20  # 滑动窗口大小
    windows = []

    for doc_words in shuffle_doc_words_list:  # 遍历每个文档/数据
        words = doc_words.split()  # 分词
        length = len(words)  # 文档包含的词数
        if length <= window_size:  # 小于窗口大小
            windows.append(words)  # 把文档的词 全部添加到windows列表中
        else:  # 长度大于窗口大小 每次添加一个窗口大小的所有单词 滑动窗口 每次右移一个单词
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}  # 统计词在多少个窗口中出现  词:包含该词的窗口数

    for window in windows:  # 遍历各个窗口
        appeared = set()
        for i in range(len(window)):  # 遍历每个窗口中的词 统计词频
            if window[i] not in appeared:  # 在一个窗口中 多次出现的词统计一次
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

    word_pair_count = {}  # 统计所有窗口中 两个词的共现次数

    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                # print(vocab[word_i_id])
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights 词节点和词节点之间的权重为PMI

    num_window = len(windows)  # 窗口数

    # 计算两个词之间的PMI
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:  # 忽略PMI为负的词对
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)  # 单词节点和单词节点之间的权重

    # word vector cosine similarity as weights

    '''
    for i in range(vocab_size):
        for j in range(vocab_size):
            if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                vector_i = np.array(word_vector_map[vocab[i]])
                vector_j = np.array(word_vector_map[vocab[j]])
                similarity = 1.0 - cosine(vector_i, vector_j)
                if similarity > 0.9:
                    print(vocab[i], vocab[j], similarity)
                    row.append(train_size + i)
                    col.append(train_size + j)
                    weight.append(similarity)
    '''
    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:  # 训练集
                row.append(i)
            else:  # 测试集
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)  # 文档节点和单词节点之间的权重 tf-idf
            doc_word_set.add(word)

    # 训练集和测试集所有的文档 每个文档对应一个节点 词典中的每个词也对应一个节点
    node_size = train_size + vocab_size + test_size
    # 用整个数据集和词典构图  adj为这个大图的邻接矩阵
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    # dump objects
    with open("./data/ind.{}.x".format(dataset), 'wb') as f:
        pkl.dump(x, f)

    with open("./data/ind.{}.y".format(dataset), 'wb') as f:
        pkl.dump(y, f)

    with open("./data/ind.{}.tx".format(dataset), 'wb') as f:
        pkl.dump(tx, f)

    with open("./data/ind.{}.ty".format(dataset), 'wb') as f:
        pkl.dump(ty, f)

    with open("./data/ind.{}.allx".format(dataset), 'wb') as f:
        pkl.dump(allx, f)

    with open("./data/ind.{}.ally".format(dataset), 'wb') as f:
        pkl.dump(ally, f)

    with open("./data/ind.{}.adj".format(dataset), 'wb') as f:
        pkl.dump(adj, f)
