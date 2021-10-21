# -*- coding: utf-8 -*-
import tensorflow as tf
from config import *
import numpy as np


class CrossDQN:
    def __init__(self):
        pass

    def create_weights(self):
        with tf.variable_scope('feature_emb_weights'):
            self.feature_emb_weights = \
                {'user_id_emb': tf.get_variable('user_id_emb', shape=[USER_VOCAB_SIZE, USER_EMB_SIZE],
                                                initializer=tf.random_normal_initializer(0.0, 0.01, seed=RANDOM_SEED)),
                 'poi_id_emb': tf.get_variable('poi_id_emb', shape=[POI_VOCAB_SIZE, POI_EMB_SIZE],
                                               initializer=tf.random_normal_initializer(0.0, 0.01, seed=RANDOM_SEED)),
                 'context_id_emb': tf.get_variable('context_id_emb', shape=[CONTEXT_VOCAB_SIZE, CONTEXT_EMB_SIZE],
                                                   initializer=tf.random_normal_initializer(0.0, 0.01,
                                                                                            seed=RANDOM_SEED))
                 }
        with tf.variable_scope('channel_weights'):
            # todo 添加weights
            self.encoder_Q_channel_weights = {}
            self.encoder_K_channel_weights = {}
            self.encoder_V_channel_weights = {}

            glorot = np.sqrt(2.0 / (4 + 1))

            for i in range(0, CHANNEL_CNT):
                self.encoder_Q_channel_weights['channel_%d' % i] = tf.get_variable(
                    name='encoder_Q_channel_%d' % i,
                    shape=[4, 4],
                    initializer=tf.random_normal_initializer(0.0, glorot),
                    dtype=tf.float32)
                self.encoder_K_channel_weights['channel_%d' % i] = tf.get_variable(
                    name='encoder_K_channel_%d' % i,
                    shape=[4, 4],
                    initializer=tf.random_normal_initializer(0.0, glorot),
                    dtype=tf.float32)
                self.encoder_V_channel_weights['channel_%d' % i] = tf.get_variable(
                    name='encoder_V_channel_%d' % i,
                    shape=[4, 4],
                    initializer=tf.random_normal_initializer(0.0, glorot),
                    dtype=tf.float32)

    def create_model(self, features):
        # IRM
        emb_map = self._emb_layer(features)
        ads_emb, ois_emb = self._target_attention(emb_map)
        # SDM
        signal_vector = self._SACU(ads_emb, ois_emb, features['action'])
        whole_action_vector = self._SACU2(ads_emb, ois_emb)
        # V, A
        V_value = self._V_network(ads_emb, ois_emb)
        A_value = self._A_network(signal_vector)
        self.Q_value = V_value + A_value
        # res los
        A_value_whole_action = tf.reshape(self._A_network(tf.reshape(whole_action_vector, [-1, 5 * 4])), [-1, 32])
        max_q_action_index = tf.reshape(tf.one_hot(tf.argmax(A_value_whole_action, axis=1), depth=32), [-1, 32])
        self.eval_res = tf.reduce_sum(tf.constant(WHOLE_ACTION_RES) * max_q_action_index, axis=[0, 1])

    def _emb_layer(self, features):
        user_emb = tf.reshape(tf.nn.embedding_lookup(
            self.feature_emb_weights['user_id_emb'], features['user_id']),
            [-1, 1, USER_EMB_SIZE])
        behavior_emb = tf.reshape(tf.nn.embedding_lookup(
            self.feature_emb_weights['poi_id_emb'], features['behavior_poi_id_list']),
            [-1, 10, POI_EMB_SIZE])
        ad_emb = tf.reshape(tf.nn.embedding_lookup(
            self.feature_emb_weights['poi_id_emb'], features['ad_id_list']),
            [-1, 5, POI_EMB_SIZE])
        oi_emb = tf.reshape(tf.nn.embedding_lookup(
            self.feature_emb_weights['poi_id_emb'], features['oi_id_list']),
            [-1, 5, POI_EMB_SIZE])
        context_emb = tf.reshape(tf.nn.embedding_lookup(
            self.feature_emb_weights['context_id_emb'], features['context_id']),
            [-1, 1, CONTEXT_EMB_SIZE])
        return {"user_emb": user_emb, "behavior_emb": behavior_emb,
                "ad_emb": ad_emb, "oi_emb": oi_emb, "context_emb": context_emb}

    def _target_attention(self, emb_map):
        ad_attention_res = self._attention(emb_map['ad_emb'], emb_map['behavior_emb'])
        oi_attention_res = self._attention(emb_map['oi_emb'], emb_map['behavior_emb'])
        ad_input_all = tf.concat([ad_attention_res, emb_map['ad_emb'], emb_map['user_emb'], emb_map['context_emb']],
                                 axis=-2)
        oi_input_all = tf.concat([oi_attention_res, emb_map['oi_emb'], emb_map['user_emb'], emb_map['context_emb']],
                                 axis=-2)
        mlp_input_all = tf.concat([ad_input_all, oi_input_all], axis=-2)
        mlp_layer_1_all = tf.layers.dense(mlp_input_all, 80, activation=tf.nn.sigmoid, name='f1_mlp',
                                          reuse=tf.AUTO_REUSE)
        mlp_layer_2_all = tf.layers.dense(mlp_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_mlp',
                                          reuse=tf.AUTO_REUSE)
        mlp_layer_3_all = tf.layers.dense(mlp_layer_2_all, 4, activation=None, name='f3_mlp',
                                          reuse=tf.AUTO_REUSE)
        return tf.gather(mlp_layer_3_all, list(range(0, 5)), axis=1), tf.gather(mlp_layer_3_all, list(range(5, 10)),
                                                                                axis=1)

    def _attention(self, ori_queries, ori_keys):
        queries = tf.tile(tf.expand_dims(ori_queries, axis=2),
                          [1, 1, ori_keys.get_shape().as_list()[1], 1])  # B * (T1 * T) * H
        keys = tf.tile(tf.expand_dims(ori_keys, axis=1),
                       [1, ori_queries.get_shape().as_list()[1], 1, 1])  # B * (T1 * T) * H
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att',
                                        reuse=tf.AUTO_REUSE)  # [B, T, 80]
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att',
                                        reuse=tf.AUTO_REUSE)  # [B, T, 40]
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att',
                                        reuse=tf.AUTO_REUSE)  # [B, T, 1]
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, tf.shape(queries)[1], tf.shape(queries)[2]])
        outputs = d_layer_3_all
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)
        # Weighted sum
        outputs = tf.matmul(outputs, ori_keys)  # [B,T1,H]
        return outputs

    def _SACU(self, ads_emb, ois_emb, action):
        ad_index = action  # batch * 5
        nature_index = 1 - action  # batch * 5
        ad_cum_index = tf.cumsum(ad_index, axis=1) - 1  # batch * 5
        nature_cum_index = tf.cumsum(nature_index, axis=1) - 1  # batch * 5
        ad_cum_index_onehot = tf.one_hot(tf.where(ad_cum_index < 0, tf.zeros_like(ad_cum_index), ad_cum_index),
                                         depth=5)  # batch * 5 * 5
        nature_cum_index_onehot = tf.one_hot(
            tf.where(nature_cum_index < 0, tf.zeros_like(ad_cum_index), nature_cum_index), depth=5)  # batch * 5 * 5
        cross_fea = tf.matmul(tf.cast(ad_cum_index_onehot, tf.float32), ads_emb) * tf.expand_dims(
            tf.cast(ad_index, tf.float32), axis=2) + tf.matmul(
            tf.cast(nature_cum_index_onehot, tf.float32), ois_emb) * tf.expand_dims(tf.cast(nature_index, tf.float32),
                                                                                    axis=2)  # batch * 5
        return tf.reshape(cross_fea, [-1, 5 * 4])

    def _SACU2(self, ads_emb, ois_emb):
        action = tf.expand_dims(tf.constant(WHOLE_ACTION), axis=0)
        ad_index = action  # 32 * 5
        nature_index = 1 - action  # 32 * 5
        ad_cum_index = tf.cumsum(ad_index, axis=2) - 1  # 32 * 5
        nature_cum_index = tf.cumsum(nature_index, axis=2) - 1  # 32 * 5
        ad_cum_index_onehot = tf.one_hot(tf.where(ad_cum_index < 0, tf.zeros_like(ad_cum_index), ad_cum_index),
                                         depth=5)  # 32 * 5 * 5
        nature_cum_index_onehot = tf.one_hot(
            tf.where(nature_cum_index < 0, tf.zeros_like(ad_cum_index), nature_cum_index),
            depth=5)  # 32 * 5 * 5
        # (b, 32, 5, 2) * (1, 32, 5, 1) + (b, 32, 5, 2) * (1, 32, 5, 1)
        cross_fea = tf.matmul(tf.cast(ad_cum_index_onehot, tf.float32),
                              tf.tile(tf.expand_dims(ads_emb, axis=1), [1, 32, 1, 1])) * tf.expand_dims(
            tf.cast(ad_index, tf.float32), axis=3) + \
                    tf.matmul(tf.cast(nature_cum_index_onehot, tf.float32),
                              tf.tile(tf.expand_dims(ois_emb, axis=1), [1, 32, 1, 1])) * tf.expand_dims(
            tf.cast(nature_index, tf.float32), axis=3)
        return tf.reshape(cross_fea, [-1, 32, 5 * 4])

    def _MCAU(self, ori_fea, channel_index):
        channel_mask = tf.expand_dims(tf.constant(CHANNEL_MASK[channel_index], dtype=tf.float32), axis=0)

        encoder_Q = tf.matmul(tf.reshape(ori_fea, (-1, 4)) * channel_mask,
                              self.encoder_Q_channel_weights['channel_%d' % channel_index])  # (batch * 5) * 4
        encoder_K = tf.matmul(tf.reshape(ori_fea, (-1, 4)) * channel_mask,
                              self.encoder_K_channel_weights['channel_%d' % channel_index])  # (batch * 5) * 4
        encoder_V = tf.matmul(tf.reshape(ori_fea, (-1, 4)) * channel_mask,
                              self.encoder_V_channel_weights['channel_%d' % channel_index])  # (batch * 5) * 4

        encoder_Q = tf.reshape(encoder_Q,
                               (tf.shape(ori_fea)[0], 5, 4))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (tf.shape(ori_fea)[0], 5, 4))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (tf.shape(ori_fea)[0], 5, 4))  # batch * 5 * 4

        # 加mask
        key_masks = tf.sign(tf.abs(tf.reduce_sum(tf.reshape(ori_fea, (-1, 5, 4)), axis=-1)))  # batch * 5
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, 5, 1])  # batch * 5 * 5

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        paddings = tf.ones_like(attention_map) * (-2 ** 32 + 1)
        attention_map = tf.where(tf.equal(key_masks, 0), paddings, attention_map)
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V), (-1, 20))  # batch * 5 * 2
        return output

    def _V_network(self, ads_emb, ois_emb):

        vnet_input = tf.reshape(tf.concat([ads_emb, ois_emb], axis=1),[-1, 40])
        vnet_mlp_layer_1_all = tf.layers.dense(vnet_input, 40, activation=tf.nn.sigmoid, name='v1_mlp',
                                               reuse=tf.AUTO_REUSE)
        vnet_mlp_layer_2_all = tf.layers.dense(vnet_mlp_layer_1_all, 20, activation=tf.nn.sigmoid, name='v2_mlp',
                                               reuse=tf.AUTO_REUSE)
        vnet_mlp_layer_3_all = tf.layers.dense(vnet_mlp_layer_2_all, 1, activation=None, name='v3_mlp',
                                               reuse=tf.AUTO_REUSE)
        return vnet_mlp_layer_3_all

    def _A_network(self, ori_fea):
        fea = ori_fea
        for i in range(CHANNEL_CNT):
            fea = tf.concat([fea, self._MCAU(ori_fea, i)], axis=1)
        anet_mlp_layer_1_all = tf.layers.dense(fea, 80, activation=tf.nn.sigmoid, name='a1_mlp',
                                               reuse=tf.AUTO_REUSE)
        anet_mlp_layer_2_all = tf.layers.dense(anet_mlp_layer_1_all, 40, activation=tf.nn.sigmoid, name='a2_mlp',
                                               reuse=tf.AUTO_REUSE)
        anet_mlp_layer_3_all = tf.layers.dense(anet_mlp_layer_2_all, 1, activation=None, name='a3_mlp',
                                               reuse=tf.AUTO_REUSE)

        return anet_mlp_layer_3_all

    def model_fn_estimator(self, features, labels, mode):
        self.create_weights()
        self.create_model(features)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.create_loss(labels)
            self.create_optimizer()
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            outputs = {'Q_value': tf.identity(self.Q_value, "Q_value")}
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                  tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)

    def create_loss(self, labels):
        self.reward = AD_WEIGHT * labels['r_ad'] + FEE_WEIGHT * labels['r_fee'] + REX_WEIGHT * labels['r_ex']
        self.loss = tf.reduce_mean(tf.square(self.Q_value - self.reward))
        if USE_AUX_RES_LOSS:
            self.loss = self.loss + AUX_RES_LOSS_WIEHGT * tf.square(self.eval_res - TARGET_RES)

    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
