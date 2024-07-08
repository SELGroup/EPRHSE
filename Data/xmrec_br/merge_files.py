import os
import gzip
import json

def merge_json_gz_files(directory, output_file):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    out = {}
    num=0
    # 遍历指定目录下的所有文件
    file_list = sorted(os.listdir(input_directory))
    for filename in file_list:
        if filename.endswith('.json.gz'):
            print(filename)
            file_path = os.path.join(directory, filename)
            with gzip.open(file_path,'rt',encoding='utf-8') as json_file:
                line = json_file.readline()
                while line != '':
                    data = json.loads(line)
                    out[num] = data
                    num+=1
                    line = json_file.readline()

    with open(output_file, 'w', encoding='utf-8') as json_file:
        for d in out.values():
            json_str = json.dumps(d)
            json_file.write(json_str + '\n')  # 写入每个字典对象，并换行
        print(f"Stored JSON objects to '{output_file}' successfully.")


def merge_txt_gz_files(input_directory, output_file):
    # 获取所有的 .txt.gz 文件
    files_to_merge = []
    out=""
    num=0
    file_list = sorted(os.listdir(input_directory))
    for filename in file_list:
        if filename.endswith('.txt.gz'):
            print(filename)
            file_path = os.path.join(input_directory, filename)
            num+=1
            with gzip.open(file_path,'rt',encoding='utf-8') as txt_file:
                for line in txt_file:
                    line = line.rstrip('\n')
                    out+=f"{line} {num-1}\n"
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(out)
        print(f"Stored merged text to '{output_file}' successfully.")


# 指定文件夹路径和输出文件名
input_directory = './Data/xmrec_br/metadata'  # 替换为实际目录路径
output_file = './Data/xmrec_br/metadata/all.json'
# 调用函数
merge_json_gz_files(input_directory, output_file)


# 指定文件夹路径和输出文件名
input_directory = './Data/xmrec_br/rating'  # 替换为实际目录路径
output_file = './Data/xmrec_br/rating/all.txt'

# 调用函数
merge_txt_gz_files(input_directory, output_file)
