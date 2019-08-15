# mental health in Tech Survey 数据来源于Kaggle


# 分析各个国家的不同性别的健康状况

import pandas as pd
import numpy as np
from pathlib import Path

pd.options.display.max_columns = 10
root = Path('.')
survey = root / '..' / u'input' / u'survey.csv'
res = root / '..' / u'output' / u'res.csv'


def parse_data():
    data = pd.read_csv(survey, sep=',')
    # Gender 这一列有数据Female,Male,M,male,m,female 等，规整化
    # print(data.describe())  # 查看基本情况
    # print('===' * 7)
    # print(data.count())  # 计数每一列的值
    # print('===' * 7)
    # print(data['Gender'].value_counts())  # 对Gender 这一列的频数统计
    # print('===' * 7)
    data['Gender'] = data['Gender'].str.lower().str.strip()  # 转化为小写并去除前后空格
    data['Gender'] = data['Gender'].str.replace('^m$', 'male', regex=True)  # 替换
    data['Gender'] = data['Gender'].replace({'^woman$': 'female', '^m$': 'male', '^f$': 'female', '^man$': 'male'},
                                            regex=True)  # 替换字典映射,不能用series.str.replace()
    print('===' * 7)
    data['Gender'] = data['Gender'].apply(lambda gender: gender if gender in ['male', 'female'] else 'other')
    print(data['Gender'].value_counts())  # 对Gender 这一列的频数统计

    # pt = data.pivot_table(index=['Country', 'Gender'], values='state', aggfunc=[np.size])     #透视图方法
    # print(pt)
    # grouped = data['Country'].groupby(data['Gender']).mean  # groupby 方法
    # size = grouped.size
    # tbl = pt.query('Gender==["male","female"]')  # 对透视表进行过滤，只能对index 进行过滤
    # tbl.sort_values(by=['state'], ascending=False)
    print(grouped)
    # tbl.to_csv(res, sep=',', index=True)


if __name__ == '__main__':
    parse_data()
