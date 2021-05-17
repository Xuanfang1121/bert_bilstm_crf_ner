# -*- coding: utf-8 -*-
# @Time    : 2021/5/15 0:17
# @Author  : zxf
import re
import json
import requests
import traceback

import numpy as np
from flask import Flask, request, jsonify

from utils import bio_to_json
from utils import get_seg_sent
from common.common import logger
from utils import split_sent_max_len
from save_file_config import load_config
from bert_data_loader import input_from_line
from context_preprocessing import count_korean_null_info
from context_preprocessing import preprocessing_korean_null
from config.ner_extract_config import get_ner_params_parser
from processing_predict_bio_label import processing_general_format

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

args = get_ner_params_parser()
config = load_config(args["config_file"])
tag2id = load_config(args["tag2id_path"])
id2tag = {value: key for key, value in tag2id.items()}


def bert_ner_model_infer(sentence, tag_to_id, id_to_tag, max_length, docker_url):
    final_result = []
    return_result = {"code": 200, "message": "success"}
    # sentence 去掉空格
    sentence_ = sentence.replace('\r\n', '✈').replace(' ', '✈')
    sentence_ = sentence_.replace('\u3000', '✈')  # todo 新增处理
    sentence_ = sentence_.replace('\xa0', '✈')
    # 统计空格和韩文的信息
    null_index_list, korean_index_list, null_korean_dict, m1 = count_korean_null_info(sentence_)
    sentence_ = ''.join(sentence_.split())
    # sentence 去掉韩文
    sentence_ = re.sub('[\uac00-\ud7ff]+', '', sentence_)
    try:
        # for sentence in text:
        # _, segment_ids, word_ids, word_mask, label_ids
        token_result = input_from_line(sentence_, max_length, tag_to_id)  # FLAGS.max_seq_len
        word_ids = token_result[2].tolist()
        word_mask = token_result[3].tolist()
        seg_ids = token_result[1].tolist()
        data = json.dumps({"signature_name": "serving_default",
                           "inputs": {"input_ids": word_ids,
                                      "input_mask": word_mask,
                                      "segment_ids": seg_ids,
                                      "dropout": 1.0}})

        headers = {"content-type": "application/json"}
        json_response = requests.post(docker_url, data=data,
                                      headers=headers)
        if json_response.status_code == 200:
            result = json.loads(json_response.text)
            if result == '':
                temp = {'content': sentence,
                        "Entity": []}
                logger.info("实体识别最终结果:" + str(temp))
                return temp
            else:
                pred = result["outputs"][0]
                pred = np.array(pred)
                label_list = pred.argmax(axis=1).tolist()[1:-1]
                # 还原空格和韩语
                if len(null_korean_dict) > 0:
                    label_list = preprocessing_korean_null(korean_index_list, null_korean_dict, m1, label_list,
                                                           tag_to_id)
                pred_label = []
                for i in range(min(len(sentence), max_length-2)):  # FLAGS.max_seq_len
                    pred_label.append(id_to_tag[label_list[i]])
                pred_label = processing_general_format(pred_label)
                logger.info("模型预测的结果: {}".format(pred_label))
                # pred_label = pred_label[1:-1]
                res = bio_to_json(sentence, pred_label)

                if len(res['entities']) != 0:
                    for i in range(len(res['entities'])):
                        if res['entities'][i]["name"] != " ":
                            final_result.append(res['entities'][i])
                else:
                    final_result = []
                temp = {'content': sentence,
                        "Entity": final_result}
                logger.info("实体识别抽取结果："+str(temp))
                return temp

        else:
            temp = {'content': sentence,
                    "Entity": []}
            logger.info("实体识别抽取最终结果:" + str(temp))
            return temp
    except Exception as e:
        logger.error(traceback.format_exc())
        return_result["code"] = 400
        return_result["message"] = traceback.format_exc()
        return_result['content'] = ''
        return_result["Entity"] = []
        return return_result


@app.route("/ner", methods=['POST'])
def ner_infer():
    data = json.loads(request.get_data(), encoding="utf-8")
    docker_url = data.get("url")
    text = data.get('context')
    print("text length: ", len(text))
    # 句子判断最大长度，如果超过最大长度，按照最大长度附近的标点符号截断
    if len(text) > args["max_seq_len"]:
        symbol_list = ["，", "；", "。"]
        split_sent = get_seg_sent(text, args["max_seq_len"], symbol_list)
        print("split_sent: ", split_sent)
        sent_entity = []
        for sent in split_sent:
            entity_result = bert_ner_model_infer(sent, tag2id, id2tag, args["max_seq_len"], docker_url)
            sent_entity.append({"sent": sent,
                                "entity": entity_result['Entity']})

        split_sent_length = [len(sent) for sent in split_sent]
        split_sent_len_sum = []
        sum_value = 0
        for i in range(1, len(split_sent_length)):
            sum_value += split_sent_length[i - 1]
            split_sent_len_sum.append(sum_value)
        sent_entity_result = [sent_entity[0]["entity"]]

        for i in range(len(split_sent_len_sum)):
            temp = []
            if len(sent_entity[i + 1]["entity"]) > 0:
                for iterm in sent_entity[i + 1]["entity"]:
                    iterm["index"] = iterm["index"] + split_sent_len_sum[i]
                    temp.append(iterm)
                sent_entity_result.extend(temp)
        result = {"content": text,
                  "Entity": sent_entity_result}

    else:
        result = bert_ner_model_infer(text, tag2id, id2tag, args["max_seq_len"], docker_url)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
