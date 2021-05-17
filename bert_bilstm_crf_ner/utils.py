# -*- coding: utf-8 -*-
import os
import json
import shutil
import logging
import codecs
import logging
import numpy as np
import tensorflow as tf


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# def test_ner(results, path):
#     """
#     Run perl script to evaluate model
#     """
#     script_file = "conlleval"
#     output_file = os.path.join(path, "ner_predict.utf8")
#     result_file = os.path.join(path, "ner_result.utf8")
#     with open(output_file, "w") as f:
#         to_write = []
#         for block in results:
#             for line in block:
#                 to_write.append(line + "\n")
#             to_write.append("\n")
#
#         f.writelines(to_write)
#     os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
#     eval_lines = []
#     with open(result_file) as f:
#         for line in f:
#             eval_lines.append(line.strip())
#     return eval_lines


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with codecs.open(output_file, "w", 'utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.exists(params.ner_bert_output_dir):
        os.makedirs(params.ner_bert_output_dir)
    if not os.path.join(params.ner_bert_summary_path):
        os.makedirs(params.ner_bert_summary_path)
    if not os.path.isdir(params.ner_bert_result_path):
        os.makedirs(params.ner_bert_result_path)
    if not os.path.isdir(params.ner_bert_ckpt_path):
        os.makedirs(params.ner_bert_ckpt_path)
    if not os.path.exists(os.path.join(params.ner_bert_output_dir, 'log')):
        os.makedirs(os.path.join(params.ner_bert_output_dir, 'log'))
    # if not os.path.isdir("log"):
    #     os.makedirs("log")


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    # if os.path.isfile(params.vocab_file):
    #     os.remove(params.vocab_file)

    if os.path.isfile(params.ner_bert_map_file):
        os.remove(params.ner_bertmap_file)

    if os.path.isdir(params.ner_bert_ckpt_path):
        shutil.rmtree(params.ner_bert_ckpt_path)

    if os.path.isdir(params.ner_bert_summary_path):
        shutil.rmtree(params.ner_nert_summary_path)

    if os.path.isdir(params.ner_bert_result_path):
        shutil.rmtree(params.ner_bert_result_path)

    # if os.path.isdir("log"):
    #     shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.ner_bert_config_file):
        os.remove(params.ner_bert_config_file)


def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def save_model(sess, model, path, global_steps):
    checkpoint_path = os.path.join(path, "model.ckpt")
    logging.info("checkpoint_path: {}".format(checkpoint_path))
    model.saver.save(sess, checkpoint_path, global_step=global_steps)
    logging.info("model saved")


def create_model(session, Model_class, path, config):
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
        #saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


def bio_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""
    # assert len(string)==len(tags), "string length is: {}, tags length is: {}".format(len(string), len(tags))

    for c_idx in range(len(tags)):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags)-1:
            tag_next = tags[c_idx+1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                # item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
                item["entities"].append({"name": c, "index": iCount, "tag": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx-1][2:] or tags[c_idx-1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    # item["entities"].append({"word": entity_name, "start": entity_start, "end": iCount + 1,
                    # "type": entity_tag})
                    item["entities"].append({"name": entity_name, "index": entity_start,
                                             "tag": entity_tag})
                    entity_name = ''
        iCount += 1
    return item


def convert_single_example(char_line, tag_to_id, max_seq_length, tokenizer, label_line):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为lb
    """
    text_list = char_line.split(' ')
    label_list = label_line.split(' ')

    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("O")   # 原来的'X'可以判断tokenizer的词是否在字典中，这里不包含对韩文的处理
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(tag_to_id["O"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(tag_to_id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(tag_to_id["O"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, input_mask, segment_ids, label_ids


def split_sent_max_len(sentence, max_len):
    result = []
    if len(sentence) > max_len:
        for i in range(0, len(sentence), max_len):
            if i + max_len > len(sentence):
                result.append(sentence[i:len(sentence)])
            else:
                result.append(sentence[i: i + max_len])
    else:
        result.append(sentence)
    return result


def get_seg_sent(sentence, max_len, symbol_list):
    symbol_index = [i for i in range(len(sentence)) if sentence[i] in symbol_list]
    print("symbol_index: ", symbol_index)
    result = []
    if len(symbol_index) <= 1:
        sent_split_result = split_sent_max_len(sentence, max_len)
        result.extend(sent_split_result)
    else:
        # 找最大长度句子处的标点符号，按照标点符号截断句子，有一个好处是避免直接按照最大长度切分句子将实体切分开的现象
        symbol_nearest_index = get_max_len_nearest_symbol(symbol_index, max_len)
        print("symbol_index_index: ", symbol_nearest_index)
        for i in range(len(symbol_nearest_index)):
            if i == 0:
                sent = sentence[: symbol_nearest_index[i] + 1]
            else:
                sent = sentence[symbol_nearest_index[i-1] + 1: symbol_nearest_index[i] + 1]
            sent_split_result = split_sent_max_len(sent, max_len)
            result.extend(sent_split_result)
    return result


def get_max_len_nearest_symbol(symbol_index, max_len):
    """确定离最大长度最近的符号"""
    # 找到第一个离最大句长最近的符号
    index = 0
    for i in range(len(symbol_index)):
        if i == 0:
            if symbol_index[i] > max_len:
                index = i
                break
        else:
            if symbol_index[i] > max_len and symbol_index[i - 1] <= max_len:
                index = i - 1
                break
    if index > 0:
        index_result = [symbol_index[index]]
    else:
        index_result = []
    # 遍历句子中所有离最大句长最近的符号
    for i in range(index + 1, len(symbol_index)):
        if i < len(symbol_index) - 1:
            if symbol_index[i] - symbol_index[index] > max_len:
                index_result.append(symbol_index[i - 1])
                index = i - 1
        else:
            if symbol_index[i] - symbol_index[index] > max_len:
                index_result.append(symbol_index[i - 1])
                index_result.append(symbol_index[i])
            elif symbol_index[i] - symbol_index[index] <= max_len:
                index_result.append(symbol_index[i])
    # else:
    #     index_result = symbol_index
    return index_result