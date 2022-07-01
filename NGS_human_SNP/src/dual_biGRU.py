import numpy as np
import utils as u
import torch
import torch.nn as nn

N_OUT = 2

class DualBiGRU(nn.Module):
    def __init__(self, hypers):
        super(DualBiGRU, self).__init__()
        self.hypers = hypers
        self.inputs = {
            # 'seq': np.zeros(None, self.hypers['n_nodes'], self.hypers['x_dim']),
            # 'global_features'  : np.zeros(None, len(self.hypers['global_feature_list'])),
            'is_training': False,
            'dropout_kp': 1.0
        }
        output_size_factor = 2
        self.mlp = u.MLP(260, 2)
        self.gru = nn.GRU(len(self.hypers['global_feature_list'])+1, self.hypers['hidden_size'], num_layers=1, bidirectional=True)
        # self.outputs = self(self.inputs)
        # self.metrics = self.make_metrics(
        #     self.outputs['pred_mean'], self.outputs['pred_precision'], self.hypers)

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

    def set_train_op(self, train_op):
        self.train_op = train_op

    def forward(self, batch):
        print("=========gru,40============")
        print(self.hypers['n_nodes']//2)
        # global_features = x[]
        x = batch['x']
        x=torch.tensor(x).float()
        print("shape")
        print(x.shape)
        
        target_seq = x[:, :self.hypers['n_nodes']//2, :]
        probe_seq = x[:, self.hypers['n_nodes']//2:, :]
        print("============(gru,45)===============")
        # print("shapes of gru input : ",target_seq.shape, len(probe_seq), target_seq[0])

        _, target_h = self.gru(target_seq)
        _, probe_h = self.gru(probe_seq)
        print("gru_first")
        print(target_h.shape)
        print(target_h.shape)
        print(probe_h.shape)
        output = torch.cat([target_h[0]+probe_h[0],
                                      target_h[1]+probe_h[1]], 1)
        print("gru_output")
        print(output.shape)
        # temp1 = output.expand(999,80,256)
        # temp2 = torch.tensor(batch['global_features'][0]).expand(999,80,4)
        print("before cat")
        print(torch.tensor(batch['global_features'][0]).shape)
        h = torch.cat([self.hypers['local_feature_weight'] * output, torch.tensor(batch['global_features'][0])], axis=1)
        mlp_output = self.mlp(h)
        print("mlp fin")
        return {
            'pred_mean': np.reshape(mlp_output[:,0], [-1]),
            'pred_precision': np.reshape(abs(mlp_output[:,1]), [-1])
        }