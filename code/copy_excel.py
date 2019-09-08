import pandas as pd
from pathlib import Path

root = Path.cwd()
input = root / '..' / 'input'
output = root / '..' / 'output'
# 获取当前目录下的所有xslx 文件,返回是个generate
files = input.glob('*.xlsx')


# files = sorted(files)  # 转成一个list


def merge_content():
    for file in files:
        content = pd.read_excel(file, sheet_name=None)  # 读取所有的sheet
        sheet_names = content.keys()  # 获取sheet 的name
        all_cont = pd.DataFrame()
        for sheet_name in sheet_names:
            cont = pd.read_excel(file, sheet_name=sheet_name)
            cont = cont.replace('"', '')
            # 读取的内容合并起来
            all_cont = pd.concat([all_cont, cont], axis=0)

        all_cont.drop_duplicates(subset='图片名', inplace=True, keep='first')
        output_name = output / 'total.csv'
        if output_name.exists():
            output_name.unlink()
        all_cont.to_csv(output_name, index=None, header=True, mode='w')


if __name__ == '__main__':
    merge_content()
