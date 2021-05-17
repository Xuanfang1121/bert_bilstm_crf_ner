# -*- coding: utf-8 -*-
# @Time    : 2021/1/27 14:09
# @Author  : zxf
import re


"""
    对预测的context进行空格和韩语的统计以及还原原文的空格和韩语
"""


def count_korean_null_info(sentence_ori):
    """
      sentence_ori: string
      null_index_list: 空格的index， list
      korean_index_list: 韩语的index， list
      null_korean_dict: 字典，key：korean_i,null_i
      m1: 韩语list
    """

    # 构建空格和韩文的字典
    null_korean_dict = dict()
    # 韩文index
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    m1 = re.findall(re_words, sentence_ori)
    korean_index_list = []
    if len(m1) > 0:
        for i in range(len(m1)):
            iterm = m1[i]
            korean_index_list.append(sentence_ori.index(iterm))
            key = "korean_" + str(i)
            null_korean_dict[key] = sentence_ori.index(iterm)

    # 统计空格
    null_index_list = []
    for i in range(len(sentence_ori)):
        if sentence_ori[i] == ' ':
            null_index_list.append(i)
            key = "null_" + str(i)
            null_korean_dict[key] = i
    return null_index_list, korean_index_list, null_korean_dict, m1


# 对去除空格和韩文后的context进行还原
def preprocessing_korean_null(korean_index_list, null_korean_dict, m1, label_list, tag2id):
    # 空格和韩文还原
    # 对字典按照values进行排序
    null_korean_dict = dict(sorted(null_korean_dict.items(), key=lambda e: e[1]))
    if len(null_korean_dict) > 0:
        for key in null_korean_dict.keys():
            index = null_korean_dict[key]
            if 'korean_' in key:
                korean_index = korean_index_list.index(index)
                for i in range(index, index + len(m1[korean_index])):
                    label_list.insert(i, tag2id['O'])
            elif 'null_' in key:
                label_list.insert(index, tag2id['O'])

    return label_list


def count_korean_info(sentence_ori):
    """
      sentence_ori: string
      null_index_list: 空格的index， list
      korean_index_list: 韩语的index， list
      null_korean_dict: 字典，key：korean_i,null_i
      m1: 韩语list
    """

    # 构建韩文的字典
    null_korean_dict = dict()
    # 韩文index
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    m1 = re.findall(re_words, sentence_ori)
    korean_index_list = []
    if len(m1) > 0:
        for i in range(len(m1)):
            iterm = m1[i]
            korean_index_list.append(sentence_ori.index(iterm))
            key = "korean_" + str(i)
            null_korean_dict[key] = sentence_ori.index(iterm)

    return korean_index_list, null_korean_dict, m1


def preprocessing_korean(korean_index_list, null_korean_dict, m1, label_list, tag2id):
    # 空格和韩文还原
    # 对字典按照values进行排序
    null_korean_dict = dict(sorted(null_korean_dict.items(), key=lambda e: e[1]))
    if len(null_korean_dict) > 0:
        for key in null_korean_dict.keys():
            index = null_korean_dict[key]
            if 'korean_' in key:
                korean_index = korean_index_list.index(index)
                for i in range(index, index + len(m1[korean_index])):
                    label_list.insert(i, tag2id['O'])
    return label_list