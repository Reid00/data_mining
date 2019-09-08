import jieba
import jieba.analyse
import jieba.posseg as pseg
from gensim import models, corpora, similarities
from collections import defaultdict
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_doc_cotent():
    # 获取文档内容
    root = Path.cwd()
    doc1 = root / '..' / 'input' / 'paper_1.txt'
    doc2 = root / '..' / 'input' / 'paper_2.txt'
    doc3 = root / '..' / 'input' / 'paper_need_compare.txt'
    # doc1, doc2 = doc1.resolve(), doc2.resolve() # 转化为绝对路径
    content1 = doc1.read_text(encoding='utf-8')
    content2 = doc2.read_text(encoding='utf-8')
    content3 = doc3.read_text(encoding='utf-8')
    return content1.strip(), content2.strip(), content3.strip()


def get_stop_words():
    # 用jieba模块进行分词并去掉无用词
    '''
       {标点符号、连词、助词、副词、介词、时语素、‘的’、数词、方位词、代词}
       {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
       去除文章中特定词性的词
       :content str
       :return list[str]
       '''
    # 特殊词性的词
    stop_flag = {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
    special_words = {'nbsp', '\u3000', '\xa0'}
    content1, content2, need_compare = get_doc_cotent()
    # 包含词性的的分词
    text1, text2 = pseg.cut(content1), pseg.cut(content2)
    texts = [list(text1), list(text2)]
    logging.info(f'词性标注后的分词为{texts}')
    # print(len(texts))
    # 获取停用词,两个文档中的停用词分开获取了，下面进行合并
    stop_words = [[word.word for word in text if word.flag in stop_flag or word.word in special_words] for text in
                  texts]
    stop_words_list = []
    for stop_word in stop_words:
        for word in stop_word:
            stop_words_list.append(word)
    stop_words_list = set(stop_words_list)
    logging.info(f'无效的停用词为{stop_words_list}')
    # 或者可以在此处直接获取除去停用词后的分词
    texts = [[word.word for word in text if word.word not in special_words and word.flag not in stop_flag] for text in
             texts]
    logging.info(f'文档分词去除停用词后为： {texts}')
    return stop_words_list


def similarity(n=2):
    stop_words = get_stop_words()
    content1, content2, need_compare_text = get_doc_cotent()
    # 分词
    texts = [list(jieba.cut(content1)), list(jieba.cut(content2))]
    # 去除停用词
    texts = [[word for word in text if word not in stop_words] for text in texts]
    logging.info(f'文档分词去除停用词后为： {texts}')
    # 计算词频
    freq = defaultdict(int)
    for text in texts:
        for token in text:
            freq[token] += 1
    # 去除词频较低的词语
    texts = [[word for word in text if freq[word] >= n] for text in texts]
    logging.info(f'文档分词去除停用词和词频较低的分词后为： {texts}')

    # 创建词袋模型
    dict = corpora.Dictionary(texts)

    # 处理要比较的文档，分词，向量化等
    need_compare_text = [word for word in jieba.cut(need_compare_text) if word not in stop_words]
    need_compare_vect = dict.doc2bow(need_compare_text)

    # 构建语料库
    corpus = [dict.doc2bow(text) for text in texts]

    # 初始化TF-IDF 模型
    tfidf = models.TfidfModel(corpus)

    # 语料库的fidf 值
    corpus_tfidf = tfidf[corpus]
    # 语料库创建索引
    index = similarities.MatrixSimilarity(corpus_tfidf)

    # 需要比对的文档tfidf 值
    need_compare_tfidf = tfidf[need_compare_vect]
    sims = index[need_compare_tfidf]
    logging.info(f'TFIDF相似度为{list(enumerate(sims))}')


if __name__ == '__main__':
    similarity()
