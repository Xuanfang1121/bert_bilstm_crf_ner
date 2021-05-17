# -*- coding: utf-8 -*-
# @Time    : 2021/5/14 18:00
# @Author  : zxf
import os
import json
import tensorflow as tf

from model import Model
from utils import create_model
from save_file_config import load_config
from bert_data_loader import input_from_line
from config.ner_extract_config import get_ner_params_parser


def predict():
    args = get_ner_params_parser()
    config = load_config(args["config_file"])
    tag2id = load_config(args["tag2id_path"])
    id2tag = {value: key for key, value in tag2id.items()}
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, args['ckpt_path'], config)
        while True:
            line = input("input sentence, please:")
            result = model.evaluate_line(sess, input_from_line(line, args["max_seq_len"], tag2id), id2tag)
            print(result['entities'])


if __name__ == "__main__":
    predict()