import os
import rarfile
import random
import sys
import logging
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
OUTPUT_PATH = os.path.join(DATA_PATH, 'output')
COMPRESSED_FILE = os.path.join(DATA_PATH, 'annie.rar')
PASSWORDS_FILE = os.path.join(DATA_PATH, 'passwords.txt')

logger = logging.getLogger(__file__)


class MyIterator():
    # 单位字符集合
    letters = 'abcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+-=<>?,./[]{}\|:"'
    min_digits = 0
    max_digits = 0

    def __init__(self, min_digits, max_digits):
        # 实例化对象时给出密码位数范围，一般4到10位
        if min_digits < max_digits:
            self.min_digits = min_digits
            self.max_digits = max_digits
        else:
            self.min_digits = max_digits
            self.max_digits = min_digits

    # 迭代器访问定义
    def __iter__(self):
        return self

    def __next__(self):
        rst = str()
        for item in range(0, random.randrange(self.min_digits, self.max_digits + 1)):
            rst += random.choice(MyIterator.letters)
        return rst


def extract(fp, password, output_path):
    try:
        fp.extractall(path=output_path, pwd=password)
        print(f"Success!!! password: {password}")
        fp.close()
        sys.exit(0)
    except:
        pass

def main(args):
    file_name = args.compressed_file
    print(f"Compressed file: {file_name}")
    print(f"Output file: {args.output_path}")
    if file_name.endswith(".rar"):
        fp = rarfile.RarFile(file_name)
    # try:
    #     passwords = open(args.passwords_file, 'r')
    # except:
    #     raise "Password file is not exist!"
    # for password in tqdm(passwords.readlines()):
    for password in MyIterator(4, 10):
        password = str(password).encode("utf-8")
        extract(fp, password, args.output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--compressed_file", type=str, default=COMPRESSED_FILE,
                        help="compressed file path.")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH,
                        help="compressed file path.")
    parser.add_argument("--passwords_file", type=str, default=PASSWORDS_FILE,
                        help="compressed file path.")
    args = parser.parse_args()
    main(args)
