from model import Model
import numpy as np
import utils as u
import torch
import torch.nn as nn

N_OUT = 2

# class DualBiGRU(Model):
#     def __init__(self, hypers):
#         super().__init__(hypers)
#         self.inputs = {
#             'seq': np.zeros(None, self.hypers['n_nodes'], self.hypers['x_dim']),
#             'global_features'  : np.zeros(None, len(self.hypers['global_feature_list'])),
#             'is_training': False,
#             'dropout_kp': 1.0
#         }
#         output_size_factor = 4 if not self.hypers['symmetric'] else 2
#         self.output_MLP = u.MLP(
#             output_size_factor*self.hypers['hidden_size'] 
#              + len(self.hypers['global_feature_list']),
#             N_OUT,
#             self.hypers['output_hid_sizes'])

#     def __call__(self, inputs, out_fmt='regression'):
#         # target_seq = inputs['seq'][:,:self.hypers['n_nodes']//2, :]
#         # probe_seq = inputs['seq'][:,self.hypers['n_nodes']//2:, :]
#         # with tf.variable_scope("target_GRU"):
#         #     _, rnn_output_states_target = nn.bidirectional_dynamic_rnn(
#         #         self.fw_cell, self.bw_cell, target_seq, dtype=tf.float32)
#         # with tf.variable_scope("probe_GRU"):
#         #     _, rnn_output_states_probe = tf.nn.bidirectional_dynamic_rnn(
#         #         self.fw_cell, self.bw_cell, probe_seq, dtype=tf.float32)
#         gru_output = nn.GRU(len(self.hypers['global_feature_list']), self.hypers['hidden_size'], num_layers=2, bidirectional=True)
        

#         # if self.hypers['bidirectional']:
#         #     fw_bw_output = tf.concat(list(rnn_output_states_target) + list(rnn_output_states_probe), 1)

#         # if self.hypers['symmetric']:
#         #     fw_bw_output = tf.concat([rnn_output_states_target[0]+rnn_output_states_probe[0],
#         #                               rnn_output_states_target[1]+rnn_output_states_probe[1]], 1)

#         if out_fmt == 'regression':
#             return self.regression_output(gru_output, inputs)
#         else:
#             return gru_output

#     def regression_output(self, h, inputs):
#         #concat global features
#         h = np.concat([self.hypers['local_feature_weight'] * h, inputs['global_features']], axis=1)
#         output = self.output_MLP(h)
#         return {
#             'pred_mean': np.reshape(output[:,0], [-1]),
#             'pred_precision': np.reshape(abs(output[:,1]), [-1])
#         }

#     def make_metrics(self, pred_mean, pred_precision, hypers):
#         # target = tf.placeholder(tf.float32, [None])
#         target=0
#         diff_sqr = (target - pred_mean)**2
#         if hypers['heteroskedastic']:
#             loss = torch.mean(0.5 * diff_sqr * pred_precision - 0.5 * torch.log(pred_precision/(2*np.pi)))
#         else:
#             loss = torch.mean(0.5 * diff_sqr)
#         accuracy = torch.sum(diff_sqr)
#         return {'loss': loss, 'accuracy': accuracy, 'target': target}

#     # def seal(self):
#     #     self.outputs = self(self.inputs)
#     #     self.metrics = self.make_metrics(
#     #         self.outputs['pred_mean'], self.outputs['pred_precision'], self.hypers)

class DualBiGRU(nn.Module):
    def __init__(self, hypers):
        self.inputs = {
            'seq': np.zeros(None, self.hypers['n_nodes'], self.hypers['x_dim']),
            'global_features'  : np.zeros(None, len(self.hypers['global_feature_list'])),
            'is_training': False,
            'dropout_kp': 1.0
        }
        output_size_factor = 4 if not self.hypers['symmetric'] else 2
        self.output_MLP = u.MLP(
            output_size_factor*self.hypers['hidden_size'] 
             + len(self.hypers['global_feature_list']),
            N_OUT,
            self.hypers['output_hid_sizes'])
        self.gru = nn.GRU(len(self.hypers['global_feature_list']), self.hypers['hidden_size'], num_layers=2, bidirectional=True)
        self.outputs = self(self.inputs)
        self.metrics = self.make_metrics(
            self.outputs['pred_mean'], self.outputs['pred_precision'], self.hypers)

    def make_metrics(self, pred_mean, pred_precision, hypers):
        # target = tf.placeholder(tf.float32, [None])
        target=0
        diff_sqr = (target - pred_mean)**2
        if hypers['heteroskedastic']:
            loss = torch.mean(0.5 * diff_sqr * pred_precision - 0.5 * torch.log(pred_precision/(2*np.pi)))
        else:
            loss = torch.mean(0.5 * diff_sqr)
        accuracy = torch.sum(diff_sqr)
        return {'loss': loss, 'accuracy': accuracy, 'target': target}

    def forward(self, x):
        h = self.gru(x)
        h = np.concat([self.hypers['local_feature_weight'] * h, x['global_features']], axis=1)
        output = self.output_MLP(h)
        return {
            'pred_mean': np.reshape(output[:,0], [-1]),
            'pred_precision': np.reshape(abs(output[:,1]), [-1])
        }