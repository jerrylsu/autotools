import re

def split_sentence(paragraph: str, flag: str = "all", limit: int = 510):
    """
    Args:
        paragraph:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zh":
            paragraph = re.sub('(?P<quotation_mark>([。？！…](?![”’"\'])))', r'\g<quotation_mark>\n',
                              paragraph)  # 单字符断句符
            paragraph = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'])', r'\g<quotation_mark>\n',
                              paragraph)  # 特殊引号
        elif flag == "en":
            paragraph = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                              paragraph)  # 英文单字符断句符
            paragraph = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', paragraph)  # 特殊引号
        else:
            paragraph = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                              paragraph)  # 单字符断句符
            paragraph = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                              paragraph)  # 特殊引号

        sent_list_ori = paragraph.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(paragraph)
    return sent_list

