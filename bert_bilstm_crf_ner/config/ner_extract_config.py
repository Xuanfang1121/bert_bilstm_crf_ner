# -*- coding: utf-8 -*-
# @Time : 2020/10/13 11:52
# @Author : zxf
import os


def get_ner_params_parser():
    base_path = os.path.dirname(__file__)
    base_path = base_path.split('config')[0]
    bert_params = dict()
    root_path = os.path.join(base_path, "model_result")
    base_upload_path = os.path.join(root_path, "pbmodel")
    bert_params["work_dir"] = root_path
    bert_params["base_upload_dir"] = base_upload_path
    bert_params["Clean"] = False
    bert_params["Train"] = True
    bert_params["Batch_size"] = 16
    bert_params["seg_dim"] = 20
    bert_params["char_dim"] = 100
    bert_params["lstm_dim"] = 256
    bert_params["num_tags"] = 10
    bert_params["tag_schema"] = 'iob'
    bert_params['clip'] = 5
    bert_params['dropout'] = 0.3
    bert_params["lr"] = 0.001
    bert_params["optimizer"] = 'adam'
    bert_params["zeros"] = False
    bert_params["lower"] = True
    bert_params["ner_type"] = 'ner'
    bert_params["max_seq_len"] = 128
    bert_params["max_epoch"] = 3
    bert_params["steps_check"] = 100
    bert_params["ckpt_path"] = os.path.join(root_path, "models")
    bert_params["pb_path"] = os.path.join(base_upload_path, "1")
    bert_params["config_file"] = os.path.join(root_path, "config.conf")
    bert_params["tag2id_path"] = os.path.join(root_path, "tag2id.dict")
    bert_params["vocab_file"] = os.path.join(root_path, "vocab.dict")
    bert_params['init_checkpoint'] = os.path.join(base_path,
                                                  "pretrain/chinese_L-12_H-768_A-12/bert_model.ckpt")
    bert_params['bert_config_file'] = os.path.join(base_path,
                                                   'pretrain/chinese_L-12_H-768_A-12/bert_config.json')
    bert_params['bert_vocab_file'] = os.path.join(base_path, 'pretrain/chinese_L-12_H-768_A-12/vocab.txt')
    bert_params["summary_path"] = os.path.join(root_path, "summary")
    bert_params["map_file"] = os.path.join(root_path, "ner_extract_words_maps.pkl")
    bert_params["vocab_file"] = "vocab.json"
    bert_params["script"] = "conlleval"
    
    return bert_params
