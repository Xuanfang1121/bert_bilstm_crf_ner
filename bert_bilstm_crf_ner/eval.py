# -*- coding: utf-8 -*-
import os

from common.common import logger
from CoNLLeval import CoNLLeval


def conlleval(label_predict, label_path, metric_path,perl_path):
    """
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    # eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(perl_path, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
        print(metrics)
    return metrics


def evaluate(sess, model, name, data, id_to_tag, tmp_eval_dir):
    # logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    word_list, tag_list, pred_list = [], [], []
    for ner_result in ner_results:
        words, tags, preds = [], [], []
        for ner in ner_result:
            ner_split = ner.split(" ")
            assert len(ner_split) == 3
            words.append(ner_split[0])
            tags.append(ner_split[1])
            preds.append(ner_split[2])
        word_list.append(words)
        tag_list.append(tags)
        pred_list.append(preds)
    ce = CoNLLeval()
    in_file = os.path.join(tmp_eval_dir, "tmp_result.txt")
    score = ce.conlleval(pred_list, tag_list, word_list, infile=in_file)
    logger.info("{} dataset -- acc: {:04.2f}, pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}"
                 .format("val", score["accuracy"], score["precision"], score["recall"], score["FB1"]))
    acc = score["accuracy"] / 100.0
    del score["accuracy"]
    # prf = score
    prf = {"p": score["precision"], "r": score["recall"], "f": score["FB1"]}
    f1 = score["FB1"]
    return acc, prf, f1
