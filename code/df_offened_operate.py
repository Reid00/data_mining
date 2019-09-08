import pandas as pd
from pathlib import Path
import datetime
import urllib.parse
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class HandleFiles:
    def __init__(self, *args):
        self.input_path = Path(args[0])
        self.output_path = Path(args[-1])
        logging.info(f'要处理的文件为{self.input_path}')

    def parse_content(self, path=None):  # 读取文本内容
        if path is None:
            path = self.input_path
        content = pd.read_csv(path, sep=',', encoding='utf-8', header=None, names=['id', 'res'])
        return content

    def rm_specific_rows(self, content, condition):  # 按条件删除指定行
        located = content.loc[content[1] == condition]  # 定位想要删除的条件
        content.drop(index=located.index, inplace=True)  # 删除满足条件的行
        return content

    def rm_dup_rows(self):  # 按删除重复行
        content = self.parse_content()
        content.drop_duplicates(subset='id', inplace=True, keep='first')
        return content

    def replace_some_content(self, content):  # 替换文本中某一列的某些字段
        pattern = r'(\?)?ref=.+'
        content[0] = content[0].apply(lambda sk: urllib.parse.unquote(sk))  # 解析url 并解码
        content[0] = content[0].apply(lambda sk: re.sub(pattern, '', sk))
        return content

    def gen_datetime_name(self):  # 生成当前时间，作为输出命名
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        return now

    def rm_dup_row_from_two_df(self, file1, file2):  # 删除两个datafram 里面某些列相同的数据
        df1 = self.parse_content(file1)
        df1 = df1[['id']]  # 拿出哪些列是需要做对比的
        df2 = self.parse_content(file2)
        df2 = df2[['id']]  # 拿出哪些列是需要做对比的
        # 不可以指定on 的条件， 两个表有相同的列名，所有会以所有的列作为on
        merged = df1.merge(df2, how='outer', indicator=True)
        # 过滤，只在左表里面有的
        left_only = merged.loc[merged['_merge'] == 'left_only']
        print(left_only.head())
        self.save_new_res(left_only)

    def save_new_res(self, content, ouput_path=None):  # 保存处理后的结果
        if ouput_path == None:
            ouput_path = self.output_path
        content.to_csv(ouput_path, sep=',', header=None, index=None, encoding='utf-8-sig')


if __name__ == '__main__':
    hf = HandleFiles(r'D:\v-baoz\python\demo\input\need_drop.csv', r'D:\v-baoz\python\demo\output\dropped.csv')
    res = hf.rm_dup_rows()
    hf.save_new_res(res)
