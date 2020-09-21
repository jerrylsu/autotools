import glob
import os

def rename_articles(file_pattern):
    """

    :param file_pattern:
    :return: 
    """
    files = glob.glob(file_pattern)
    for file_ in files:
        print(file_)
        raise "Test..."
        date = ''
        with open(file_, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        for i, line in enumerate(lines):
            if line.startswith('Date: '):
                date = line.split()[1]
                break
