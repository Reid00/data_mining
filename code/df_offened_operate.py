import pandas as pd
from pathlib import Path
import datetime
import urllib.parse
import re


class HandleFiles:
    def __init__(self, *args):
        self.input_path = Path(args[0])
        self.output_path = Path(args[-1])

    def parse_content(self):  # 读取文本内容
        content = pd.read_csv(self.input_path, sep=',', encoding='utf-8', header=None, names=['id', 'res'])
        return content

    def rm_specific_rows(self, content, condition):  # 按条件删除指定行
        located = content.loc[content[1] == condition]  # 定位想要删除的条件
        content.drop(index=located.index, inplace=True)  # 删除满足条件的行
        return content

    def rm_dup_rows(self, content):  # 按删除重复行
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

    def save_new_res(self, content):  # 保存处理后的结果
        content.to_csv(self.output_path, sep=',', header=None, index=None, encoding='utf-8-sig')


if __name__ == '__main__':
    hf = HandleFiles(r'D:\v-baoz\python\demo\input\total.csv', r'D:\v-baoz\python\demo\output\new_res.csv')
    ori_cont = hf.parse_content()
    new_res = hf.rm_dup_rows(ori_cont)
    hf.save_new_res(new_res)
