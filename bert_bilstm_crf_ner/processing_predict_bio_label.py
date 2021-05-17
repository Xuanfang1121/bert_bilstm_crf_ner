# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 11:16
# @Author  : zxf

"""
    修改模型预测的label，后处理
    修改格式：'O', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'
            'B-Y', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'
            'I-X', 'I-Y', 'I-X'
            'B-X', 'O', 'I-X'
            'B-X', 'B-X', 'B-X'
            'B-X', 'I-Y', 'I-X'
"""


def processing_general_format_xyx(pred_label):
    """
       修改：'I-X', 'I-Y', 'I-X' or 'I-X', 'O', 'I-X'
            'I-X', 'I-X', 'I-X'
    """
    for i in range(1, len(pred_label)-1):
        if pred_label[i-1] != pred_label[i] and pred_label[i+1] == pred_label[i-1] \
                and 'I-' in pred_label[i-1] and 'I-' in pred_label[i]:
            pred_label[i] = pred_label[i+1]
    return pred_label


def processing_general_format_byxx(pred_label):
    """
       修改: 'O', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'
            'B-Y', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'
       为：
          'B-X', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'
          'B-X', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'
    """
    for i in range(1, len(pred_label)-1):
        if pred_label[i] == pred_label[i+1] and pred_label[i-1] != pred_label[i] and 'I-' in pred_label[i]:
            if 'B-' in pred_label[i-1]:
                # _, tag = pred_label[i-1].split('-')
                # _, tag_ = pred_label[i].split('-')
                # if tag != tag_:
                #     pred_label[i-1] = 'B-' + tag_
                temp_tag = pred_label[i-1].split('-')
                temp_tag_ = pred_label[i].split('-')
                if len(temp_tag_) == len(temp_tag) and len(temp_tag_) == 2:
                    if temp_tag[1] != temp_tag_[1]:
                        pred_label[i - 1] = 'B-' + temp_tag_[1]
                elif len(temp_tag_) != len(temp_tag) and len(temp_tag_) == 3:
                    pred_label[i - 1] = 'B-' + temp_tag_[1] + "-" + temp_tag_[2]
        elif pred_label[i] == pred_label[i+1] and pred_label[i-1] == 'O' and 'I-' in pred_label[i]:
            # _, tag = pred_label[i].split('-')
            # pred_label[i-1] = 'B-' + tag
            temp = pred_label[i].split('-')
            if len(temp) == 2:
                pred_label[i - 1] = 'B-' + temp[1]
            elif len(temp) == 3:
                pred_label[i - 1] = 'B-' + temp[1] + '-' + temp[2]
    return pred_label


def processing_general_format_oxx(pred_label):
    """
       修改: 'O', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'

       为：
          'B-X', 'I-x', 'I-x', 'I-x', 'I-x', 'I-x'

    """
    for i in range(1, len(pred_label)-1):
        if pred_label[i] == pred_label[i+1] and pred_label[i-1] == 'O' and 'I-' in pred_label[i]:
            temp = pred_label[i].split('-')
            if len(temp) == 2:
                # _, tag = pred_label[i].split('-')
                # pred_label[i-1] = 'B-' + tag
                pred_label[i - 1] = 'B-' + temp[1]
            elif len(temp) == 3:
                # 事件抽取的触发词
                pred_label[i - 1] = 'B-' + temp[1] + '-' + temp[2]
    return pred_label


def processing_general_format_bxx(pred_label):
    """
       修改: 'B-X', 'B-X', 'B-X'

       为：
          'B-X', 'I-X', 'I-X'

    """
    index_list = []
    for i in range(1, len(pred_label)):
        if pred_label[i] == pred_label[i-1] and 'B-' in pred_label[i]:
            start_index = i - 1
            end_index = i
            for j in range(start_index+1, len(pred_label)):
                if pred_label[j] == pred_label[i]:
                    end_index = j
                else:
                    break

            if end_index > start_index:
                index_list.append([start_index, end_index])

    if len(index_list) > 0:
        for iterm in index_list:
            # _, tag = pred_label[iterm[0]].split('-')
            # for i in range(iterm[0] + 1, iterm[1]+1):
            #     pred_label[i] = 'I-' + tag
            temp = pred_label[iterm[0]].split('-')
            if len(temp) == 2:
                for i in range(iterm[0] + 1, iterm[1] + 1):
                    pred_label[i] = 'I-' + temp[1]
            elif len(temp) == 3:
                for i in range(iterm[0] + 1, iterm[1] + 1):
                    pred_label[i] = 'I-' + temp[1] + "-" + temp[2]

    return pred_label


def processing_general_format_bxox(pred_label):
    """
       修改： B-Y O I-Y or 'B-Y', 'I-X', 'I-Y'

       为：B-Y I-Y I-Y
    """
    for i in range(1, len(pred_label)-1):
        if pred_label[i] == 'O' and 'I-' in pred_label[i+1] and 'B-' in pred_label[i-1]:
            # _, tag = pred_label[i+1].split('-')
            # _, tag_ = pred_label[i-1].split('-')
            # if tag == tag_:
            #     pred_label[i] = "I-" + tag
            temp_tag = pred_label[i+1].split('-')
            temp_tag_ = pred_label[i-1].split('-')
            if len(temp_tag) == len(temp_tag_) and len(temp_tag_) == 2:
                if temp_tag[1] == temp_tag_[1]:
                    pred_label[i] = "I-" + temp_tag[1]
            elif len(temp_tag) == len(temp_tag_) and len(temp_tag_) == 3:
                if temp_tag[2] == temp_tag_[2]:
                    pred_label[i] = "I-" + temp_tag[1] + "-" + temp_tag[2]
        elif 'I-' in pred_label[i] and 'I-' in pred_label[i+1] \
                and 'B-' in pred_label[i-1] and pred_label[i] != pred_label[i+1]:
            # _, tag = pred_label[i+1].split('-')
            # _, tag_ = pred_label[i-1].split('-')
            # if tag == tag_:
            #     pred_label[i] = "I-" + tag
            temp_tag = pred_label[i + 1].split('-')
            temp_tag_ = pred_label[i - 1].split('-')
            if len(temp_tag) == len(temp_tag_) and len(temp_tag_) == 2:
                if temp_tag[1] == temp_tag_[1]:
                    pred_label[i] = "I-" + temp_tag[1]
            elif len(temp_tag) == len(temp_tag_) and len(temp_tag_) == 3:
                if temp_tag[2] == temp_tag_[2]:
                    pred_label[i] = "I-" + temp_tag[1] + "-" + temp_tag[2]
    return pred_label


def processing_general_format(pred_label):
    pred_label = processing_general_format_xyx(pred_label)
    pred_label = processing_general_format_byxx(pred_label)
    pred_label = processing_general_format_oxx(pred_label)
    pred_label = processing_general_format_bxx(pred_label)
    pred_label = processing_general_format_bxox(pred_label)
    return pred_label