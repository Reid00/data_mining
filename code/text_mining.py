import jieba
import jieba.posseg
import jieba.analyse
import pandas as pd


def cut_exercise():
    sentence = '我喜欢苏州的苏州中心,上海,上海的东方明珠'
    # 进行分词, cut_all 设置分词模式，true 全模式, 但是会叠加
    # cut_all=False 是精准模式
    words = jieba.cut(sentence, cut_all=False)  # generate
    print(list(words))
    print('======' * 10)
    # 搜索引擎的模式的切分，cut_for_search
    words2 = jieba.cut_for_search(sentence)  # generate
    print(list(words2))


# 词性的的获取
def posseg():
    sentence = '我喜欢苏州的苏州中心,上海,上海的东方明珠'
    # flag 词性，word 词语
    words = jieba.posseg.cut(sentence)
    # for word in words:
    #     print(f'flag: {word.flag}\tword: {word.word}')

    # 词典的加载,加载后就会使用本词典
    # dict_path = ''
    # jieba.load_userdict(dict_path)
    # words = jieba.cut(sentence)
    # 更改词频1. suggest_freq
    words = jieba.cut(sentence)
    print(f'更改词频前的分词 {list(words)}')
    jieba.suggest_freq('苏州中心,', True)
    words = jieba.cut(sentence)
    print(f'更改词频后的分词 {list(words)}')


# 返回词频较高的词语，即是关键词
def high_freq_words():
    sentence = '我喜欢苏州的苏州中心,上海,上海的东方明珠'
    words = jieba.analyse.extract_tags(sentence, topK=3)
    print(f'top 3 的词语 {words}')
    # 返回词语的位置
    words_loc = jieba.tokenize(sentence)
    print(f'各个词语的位置{list(words_loc)}')


# 盗墓笔记的关键词提取
def analyse_article():
    input = r'D:\v-baoz\python\data_mining\input\article.txt'
    # f = pd.read_csv(input, sep=' ', encoding='gbk', header=None, error_bad_lines=False)
    with open(input, 'rb', ) as f:
        words = jieba.analyse.extract_tags(f.read().decode('gbk').replace('&quot', '').replace('......', ''), topK=10)
        print(f'top 10 的词语{words}')


if __name__ == '__main__':
    # cut_exercise()
    # posseg()
    # high_freq_words()
    analyse_article()
