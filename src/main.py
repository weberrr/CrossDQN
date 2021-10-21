# -*- coding: utf-8 -*-
from input import *
from model import *

def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        save_summary_steps=100,
        save_checkpoints_steps=100,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=100,
        session_config=session_config
    )
    model = CrossDQN()
    estimator = tf.estimator.Estimator(model_fn=model.model_fn_estimator, config=config)
    return estimator


def save_estimator(estimator, export_dir):
    def _serving_input_receiver_fn():
        receiver_tensors = {
            'user_id': tf.placeholder(tf.int64, [None, 1], name='user_id'),
            'behavior_poi_id_list': tf.placeholder(tf.int64, [None, 10], name='behavior_poi_id_list'),
            'ad_id_list': tf.placeholder(tf.float32, [None, 5], name='ad_id_list'),
            'oi_id_list': tf.placeholder(tf.float32, [None, 5], name='oi_id_list'),
            'context_id': tf.placeholder(tf.float32, [None, 1], name='context_id'),
            'action':  tf.placeholder(tf.float32, [None, 5], name='action')
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)
    export_dir = estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=_serving_input_receiver_fn)
    return export_dir


if __name__ == '__main__':
    estimator = create_estimator()
    train_input_fn = input_fn_maker(DATA_PATH)
    estimator.train(train_input_fn)
    save_estimator(estimator, PB_SAVE_PATH)