# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from config import *


def demo_tfrecord_data(tfrecord_file_name):
    # user_id, history_poi_id_list, ad_id_list, oi_id_list, context_id, R_ad, R_fee, R_ex
    demo_data = [
        [[121, 123, 456, 789, 321, 0, 0, 0, 0, 0, 0, 23, 31, 42, 22, 67, 76, 321, 36, 93, 22, 13, 1, 2, 3, 51, 2],
         [1.2, 2, 2.1]],
        [[121, 123, 456, 789, 321, 0, 0, 0, 0, 0, 0, 23, 31, 42, 22, 67, 76, 321, 36, 93, 22, 13, 1, 2, 3, 32, 2],
         [1.2, 2, 2.1]]
    ]
    writer = tf.io.TFRecordWriter(tfrecord_file_name)
    for item in demo_data:
        exam = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=item[0])),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=item[1]))
                }
            )
        )
        writer.write(exam.SerializeToString())
    writer.close()
    print("finish to write data to tfrecord file!")


def generate_parse_tfrecord_local_fn():
    def _parse_function(batch_examples):
        data_description = {
            "feature": tf.FixedLenFeature([FEATURE_LEN], dtype=tf.int64),
            "label": tf.FixedLenFeature([LABEL_LEN], dtype=tf.float32)
        }
        parsed_features = tf.parse_example(
            batch_examples,
            features=data_description
        )
        feature_buffer = parsed_features['feature']
        features = {
            'user_id': tf.cast(tf.gather(feature_buffer, list(range(0, 1)), axis=1), tf.int64),
            'behavior_poi_id_list': tf.cast(tf.gather(feature_buffer, list(range(1, 11)), axis=1), tf.int64),
            'ad_id_list': tf.cast(tf.gather(feature_buffer, list(range(11, 16)), axis=1), tf.int64),
            'oi_id_list': tf.cast(tf.gather(feature_buffer, list(range(16, 21)), axis=1), tf.int64),
            'context_id': tf.cast(tf.gather(feature_buffer, list(range(21, 22)), axis=1), tf.int64),
            'action': tf.cast(tf.gather(feature_buffer, list(range(22, 27)), axis=1), tf.int32)
        }
        label_buffer = parsed_features['label']
        labels = {
            'r_ad': tf.gather(label_buffer, list(range(1)), axis=1),
            'r_fee': tf.gather(label_buffer, list(range(1, 2)), axis=1),
            'r_ex': tf.gather(label_buffer, list(range(2, 3)), axis=1)
        }
        return features, labels

    return _parse_function


def input_fn_maker(file_names):
    def input_fn():
        _parse_fn = generate_parse_tfrecord_local_fn()
        files = tf.data.Dataset.list_files(file_names)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4 * 10))
        dataset = dataset.prefetch(buffer_size=BATCH_SIZE * 10)
        dataset = dataset.repeat(EPOCH)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(_parse_fn, num_parallel_calls=NUM_PARALLEL)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


if __name__ == '__main__':
    # demo_tfrecord_data("part-r-00001")
    train_input_fn = input_fn_maker(DATA_PATH)
    features, labels = train_input_fn()
    sess = tf.Session()
    try:
        features_np, labels_np = sess.run([features, labels])
        print("*" * 100, "features_np")
        for key in features_np:
            print("=" * 50, key, np.shape(features_np[key]))
            print(features_np[key])
        print("*" * 100, "labels_np")
        for key in labels_np:
            print("=" * 50, key, np.shape(labels_np[key]))
            print(labels_np[key])
    except Exception as e:
        print(e)
