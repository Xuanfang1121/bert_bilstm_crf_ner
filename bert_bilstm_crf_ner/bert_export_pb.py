# -*- coding: utf-8 -*-
# @Time : 2020/10/13 11:52
# @Author : zxf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model import Model
from common.common import logger
from utils import create_model
from save_file_config import load_config
from config.ner_extract_config import get_ner_params_parser


def export_pb_file(args):
    config = load_config(args['config_file'])
    logger.info("config path: {}".format(args['config_file']))
    if not os.path.exists(args['pb_path']):
        os.makedirs(args['pb_path'])

    # files = os.listdir(FLAGS.pb_path)
    # if len(files) == 0:
    #     max_version = 1
    # else:
    #     files = list(map(int, files))
    #     max_version = max(files) + 1

    export_path = args['pb_path']
    logger.info("pb 模型路径为: {}".format(args["pb_path"]))
    # limit GPU memory
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    graph = tf.Graph()
    with graph.as_default():
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, args['ckpt_path'], config)
            input_ids = model.input_ids
            input_mask = model.input_mask
            segment_ids = model.segment_ids
            dropout = model.dropout
            logits = model.logits
            saver = tf.train.Saver()

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt_file = tf.train.latest_checkpoint(args['ckpt_path'])
            saver.restore(sess, ckpt_file)

            model_tensor_input_ids = tf.compat.v1.saved_model.utils.build_tensor_info(input_ids)
            model_tensor_input_dropout = tf.compat.v1.saved_model.utils.build_tensor_info(dropout)
            model_tensor_input_mask = tf.compat.v1.saved_model.utils.build_tensor_info(input_mask)
            model_tensor_segment_ids = tf.compat.v1.saved_model.utils.build_tensor_info(segment_ids)
            model_tensor_output = tf.compat.v1.saved_model.utils.build_tensor_info(logits)

            prediction_signature = (
                tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_ids': model_tensor_input_ids,
                            'input_mask': model_tensor_input_mask,
                            'segment_ids': model_tensor_segment_ids,
                            "dropout": model_tensor_input_dropout},
                    outputs={'predictions': model_tensor_output},
                    method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict':
                        prediction_signature,
                    tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save(as_text=False)
            logger.info('模型转pb完成')


if __name__ == "__main__":
    args = get_ner_params_parser()
    export_pb_file(args)