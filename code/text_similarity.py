# （1）使用TF-IDF算法，找出两篇文章的关键词；某个词对文章的重要性越高，它的TF-IDF值就越大
# （2）每篇文章各取出若干个关键词（比如20个），合并成一个集合，计算每篇文章对于这个集合中的词的词频（为了避免文章长度的差异，可以使用相对词频）；
# （3）生成两篇文章各自的词频向量；
# （4）计算两个向量的余弦相似度，值越大就表示越相似。

from gensim import models, corpora, similarities
import jieba
from collections import defaultdict
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 准本数据，9篇文档
documetns = [

    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]


def stop_words():
    # 设置停用词
    stop_list = set('for a of the and to in'.split())
    return stop_list


def similarity_compare(n=2):  # n 表示去除词频比较低的单词
    stop_list = stop_words()
    # 1 用空格分词，并去除停用词
    texts = [[word for word in doc.lower().split() if word not in stop_list] for doc in documetns]

    # 2 计算每个词的词频TF
    freq = defaultdict(int)  # 构建字典对象
    # 遍历分词的结果集，计算词频
    for text in texts:
        for token in text:
            freq[token] += 1
    # 选择词频大于n 的分词
    texts = [[word for word in text if freq[word] > n] for text in texts]
    logging.info(f'在每个文档里筛选出词频大于n 的单词 {texts}')

    # 3 通过corpora 创建词典 词袋模型
    # 创建词典，单词和编号之间的映射
    dict = corpora.Dictionary(texts)
    # 可以保存词典以后使用
    # dict.save(Path.cwd())
    logging.info(f'初始词典为{dict}')
    # 打印词典，key 为单词，value 为单词的编号
    logging.info(f'打印字典{dict.token2id}')
    # 从log结果可以看出每个单词都有一个编号，总共有12 个单词

    # 4 处理要比较的文档，比较的文档转换为稀疏向量
    # 文档分词，并去除停用词
    need_compare_doc = 'Human computer computer interaction'
    need_compare_text = [word for word in need_compare_doc.lower().split() if word not in stop_list]
    # 将文档分词并使用doc2bow方法对每个不同单词的词频进行了统计，并将单词转换为其在字典中的编号，然后以稀疏向量的形式返回结果
    vector = dict.doc2bow(need_compare_text)
    logging.info(f'需要对比的文档转化为稀疏向量后为{vector}')

    # 5 建立语料库
    # 将每一篇分词后的初始文档都转化为向量
    corpus = [dict.doc2bow(text) for text in texts]
    logging.info(f'初始文档建立的语料库稀疏向量为{corpus}')

    # 6 初始化模型
    # 初始化一个TFIDF model,用现有的语料库进行模型的训练，表示方法为新的表示方法（Tfidf 实数权重）
    tfidf = models.TfidfModel(corpus)
    logging.info(f'初始化的tfidf 模型为{tfidf}')
    # 将语料库转化为dfidf 表示
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        logging.info(doc)
    # 7 创建索引,使用上一步得到的带有tfidf值的语料库建立索引
    index = similarities.MatrixSimilarity(corpus_tfidf)
    logging.info(f'创建索引后的结果为{index}')

    # 8 计算相似度
    new_vec_tfidf = tfidf[vector]  # 将要比较文档转换为tfidf表示方法
    logging.info(f'需要比较的文档tfidf 值为{new_vec_tfidf}')
    # 计算要比较的文档与语料库中每篇文档的相似度
    sims = index[new_vec_tfidf]
    logging.info(f'相似度为{sims}')


if __name__ == '__main__':
    similarity_compare(1)
