# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import json
# import logger
import traceback

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from model import Model
from eval import evaluate
from common.common import logger
from bert_data_utils import BatchManager
from save_file_config import save_config
from bert_data_loader import load_sentences
from bert_data_loader import prepare_dataset
from utils import create_model, save_model
from config.ner_extract_config import get_ner_params_parser


def train(args, train_data, test_data):
    try:
        # load data sets
        train_sentences, tags = load_sentences(args['lower'], args['zeros'], train_data)
        dev_sentences, _ = load_sentences(args['lower'], args['zeros'], test_data)

        # bert model update tag, add bert two tag: '[CLS]', '[SEP]'
        tag2id = {value: key for key, value in enumerate(tags)}
        id_to_tag = {value: key for key, value in tag2id.items()}
        args["num_tags"] = len(tag2id)
        if not os.path.exists(args["work_dir"]):
            os.makedirs(args["work_dir"])
        with open(args["tag2id_path"], "w", encoding="utf-8") as f:
            f.write(json.dumps(tag2id, ensure_ascii=False))

        # save vocab
        vocab_list = []
        with open(args['bert_vocab_file'], "r", encoding="utf-8") as f:
            for line in f.readlines():
                vocab_list.append(line)
        with open(args['vocab_file'], "w", encoding="utf-8") as f:
            for line in vocab_list:
                f.write(line)

        # prepare data, get a collection of list containing index
        train_data = prepare_dataset(train_sentences, args['max_seq_len'], tag2id, args['lower'])
        dev_data = prepare_dataset(dev_sentences, args['max_seq_len'], tag2id, args['lower'])
        # logger.info("%i / %i / sentences in train / dev." % (len(train_data), len(dev_data)))

        train_manager = BatchManager(train_data, args['Batch_size'])
        dev_manager = BatchManager(dev_data, args['Batch_size'])
        # save bert model config
        save_config(args, args['config_file'])
        logger.info("模型参数保存:{}".format(args["config_file"]))
        # path check
        if not os.path.exists(args["ckpt_path"]):
            os.makedirs(args["ckpt_path"])
        # make_path(FLAGS)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        best_f1 = 0
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, args['ckpt_path'], args)
            loss = []
            for i in range(args['max_epoch']):
                epoch_loss = []
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    epoch_loss.append(batch_loss)
                    loss.append(batch_loss)
                    if step % args['steps_check'] == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                     "event loss:{:>9.6f}".format(
                                                                iteration, step % steps_per_epoch,
                                                                steps_per_epoch, np.mean(loss)))
                        loss = []

                acc, prf, f1 = evaluate(sess, model, "dev", dev_manager, id_to_tag, args["work_dir"])
                loss_result = np.mean(epoch_loss).item()
                if np.isnan(loss_result):
                    loss_result = 0.0
                if np.isnan(acc):
                    acc = 0.0
                logger.info("epoch: {}/{} dev data result: acc :{} loss: {}".format(i, args['max_epoch'],
                                                                                    acc, loss_result))
                if f1 >= best_f1:
                    best_f1 = f1
                    save_model(sess, model, args['ckpt_path'], global_steps=step)
                else:
                    continue
            return prf
    except Exception as e:
        logger.error(traceback.format_exc())
        prf = {"p": 0.0, "r": 0.0, "f": 0.0}
        return prf


def read_data(data_file):
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                data.append(" ")
            else:
                data.append(line.strip())
    return data


if __name__ == "__main__":
    args = get_ner_params_parser()
    # train_data , test_data
    train_data = read_data("./data/ner_train.data")
    test_data = read_data("./data/ner_val.data")
    prf = train(args, train_data, test_data)