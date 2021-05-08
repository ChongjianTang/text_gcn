from nltk.corpus import stopwords
import nltk  # 导入nltk 英文文本处理库
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn  # 导入wordnet

import sys

if __name__ == '__main__':
    from utils.utils import clean_str, loadWord2Vec

    if len(sys.argv) != 2:
        sys.exit("Use: python remove_words.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']  # 全部5个数据集
    dataset = sys.argv[1]  # 获取待处理的数据集名称

    if dataset not in datasets:  # 数据集名称设置错误
        sys.exit("wrong dataset name")

    """
    由于一些常用字或者词使用的频率相当的高，英语中比如a，the, he等，中文中比如：我、它、个等，
    每个页面几乎都包含了这些词汇，如果搜索引擎它们当关键字进行索引，那么所有的网站都会被索引，
    而且没有区分度，所以一般把这些词直接去掉，不可当做关键词。
    """
    nltk.download('stopwords')  # 下载停止词
    stop_words = set(stopwords.words('english'))  # 选择英文停止词 去重 得到英文停止词表
    # print(stop_words)

    # Read Word Vectors
    # word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
    # vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
    # word_embeddings_dim = len(embd[0])
    # dataset = '20ng'

    doc_content_list = []
    # with open('data/wiki_long_abstracts_en_text.txt', 'r') as f:
    # 读取相关数据集文件
    with open('./data/corpus/' + dataset + '.txt', 'rb') as f:
        for line in f.readlines():  # 按行处理 一行代表一条数据/文本
            doc_content_list.append(line.strip().decode('latin1'))  # 去除首尾空白符 用latin1解码 保存在doc_content_list列表中

    word_freq = {}  # to remove rare words 词频字典 词到频数的映射

    for doc_content in doc_content_list:
        # 数据清洗
        temp = clean_str(doc_content)
        # 按空格切分、分词
        words = temp.split()
        for word in words:  # 统计词频
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # 去除停止词和低频词(<5)
    clean_docs = []
    for doc_content in doc_content_list:
        # 数据清洗
        temp = clean_str(doc_content)
        # 按空格切分、分词
        words = temp.split()
        doc_words = []  # 保存数据集中的单词
        for word in words:
            # 如果是mr数据集 不去除停止词和低频词(<5) 文本很短
            if dataset == 'mr':
                doc_words.append(word)
            # 其他数据集 需要去除停止词和低频词(<5)
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()  # 对每条数据处理完后 用' '连接 并去除首尾空白符
        # if doc_str == '':
        # doc_str = temp
        clean_docs.append(doc_str)  # 把处理完后的每条数据/文本保存在clean_docs列表中

    clean_corpus_str = '\n'.join(clean_docs)  # 把处理完后的每条数据(列表中的每个元素)用\n连接起来

    # with open('./data/wiki_long_abstracts_en_text.clean.txt', 'w') as f:
    # 把处理好的数据集 写入文件 加后缀.clean 以作区分
    with open('./data/corpus/' + dataset + '.clean.txt', 'w') as f:
        f.write(clean_corpus_str)

    # dataset = '20ng'
    min_len = 10000
    aver_len = 0
    max_len = 0

    # with open('./data/wiki_long_abstracts_en_text.txt', 'r') as f:
    # 读取处理好的数据集
    with open('./data/corpus/' + dataset + '.clean.txt', 'r') as f:
        lines = f.readlines()  # 把所有行读到一个列表中  每行代表一条数据/文本 对应列表中的一个元素
        for line in lines:  # 按行处理
            line = line.strip()
            temp = line.split()
            # 统计数据集中 每条数据/文本的最大长度/最小长度
            aver_len = aver_len + len(temp)
            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)
    # 计算数据集中所有数据/文本的平均长度
    aver_len = 1.0 * aver_len / len(lines)
    print('Min_len : ' + str(min_len))
    print('Max_len : ' + str(max_len))
    print('Average_len : ' + str(aver_len))
