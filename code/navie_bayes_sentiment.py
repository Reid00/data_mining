"""
场景是情感分析(尤其是褒贬判定)，
Kaggle比赛之影评与观影者情感判定
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import logging
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data(path):
    data = pd.read_csv(path, sep='\t')
    logging.info(f'前五条数据为\n{data.head()}')
    logging.info(f'整体数据的描述为\n{data.describe()}')
    logging.info(f'整体数据的nan值情况为\n{data.isnull().sum()}')
    return data


def review_to_wordlist(review):
    """
    把IMDB的评论转成词序列
    :param review:
    :return:
    """
    # 去掉HTML标签，拿到内容
    review = re.sub(r'<br />', '', review)
    # 用正则表达式取出符合规范的部分
    review = re.sub('[^a-zA-Z]', ' ', review)
    # 小写化所有的词，并转成词list
    words = review.lower().split()
    return words


def handle_text(df):
    # 将训练和测试数据都转成词list
    words = []
    for i in range(len(df['review'])):
        words.append(' '.join(review_to_wordlist(df['review'][i])))
    return words


def feature_tfidf(train_words, test_words):
    """
    建立tfidf 特征向量
    :param words:
    :return:
    """
    # 初始化TFIV对象，去停用词，加2元语言模型
    tfv = TFIV(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
               ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    # 合并训练和测试集以便进行TFIDF向量化操作
    X_all = train_words + test_words
    len_train = len(train_words)
    # 这一步有点慢，train model
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)
    # 恢复成训练集和测试集部分
    X = X_all[:len_train]
    X_test = X_all[len_train:]
    return X, X_test


def train_model(X, y_train):
    """
    模型训练
    :return:
    """
    model_NB = MNB()
    model_NB.fit(X, y_train)  # 特征数据直接灌进来
    MNB(alpha=1.0, class_prior=None, fit_prior=True)
    score = np.mean(cross_val_score(model_NB, X, y_train, cv=20, scoring='roc_auc'))
    logging.info(f'多项式贝叶斯分类器20折交叉验证得分{score}')


if __name__ == '__main__':
    root = Path.cwd().parent
    train_path = root / 'input' / 'word2vec-nlp-tutorial' / 'labeledTrainData.tsv'
    test_path = root / 'input' / 'word2vec-nlp-tutorial' / 'testData.tsv'
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    # 取出情感标签，positive/褒 或者 negative/贬
    y_train = train_data['sentiment']
    train_words = handle_text(train_data)
    logging.info(f'训练集words\n{train_words[:20]}')
    test_words = handle_text(test_data)
    logging.info(f'测试集words\n{test_words[:20]}')
    X, X_test = feature_tfidf(train_words, test_words)
    train_model(X, y_train)
