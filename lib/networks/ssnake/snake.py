import torch.nn as nn
import torch
import numpy as np

class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input, adj):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input, adj):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation) # use CircConv/DilatedCircConv
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj=None):
        x = self.conv(x, adj)
        x = self.relu(x)
        x = self.norm(x)
        return x
class BasicConv(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicConv, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation) # use CircConv/DilatedCircConv

    def forward(self, x, adj=None):
        x = self.conv(x, adj)
        return x

class Snake_Attention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_queries, h=3, dropout=0.1):
        super(Snake_Attention, self).__init__()
        d_keys = d_queries
        d_k = d_queries
        d_v = d_queries
        d_out = d_queries
        self.fc_q = nn.Linear(d_queries, d_k * h)
        self.fc_k = nn.Linear(d_keys, d_k * h)
        self.fc_v = nn.Linear(d_keys, d_v * h)
        self.fc_o = nn.Linear(d_v * h, d_out)
        self.dropout = nn.Dropout(dropout)

        self.d_queries = d_queries
        self.d_keys = d_keys
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.h = h

    def forward(self, input):
        # queries:[Ninst,Nfeat,Nq]
        # keys:[Ninst,Nfeat,Nk]
        queries = input
        keys = input
        values = input
        assert len(queries.shape) == 3 and len(keys.shape) == 3
        queries = queries.permute(0, 2, 1)
        keys = keys.permute(0, 2, 1)
        values = values.permute(0, 2, 1)
        B, Nq = queries.shape[:2]
        Nk = keys.shape[1]
        q = self.fc_q(queries).view(B, Nq, self.h, self.d_k).permute(0, 2, 1, 3)  # shape[B,h,Nq,d_k]
        k = self.fc_k(keys).view(B, Nk, self.h, self.d_k).permute(0, 2, 3, 1)  # shape[B,h,d_k,Nk]
        v = self.fc_v(values).view(B, Nk, self.h, self.d_v).permute(0, 2, 1, 3)  # shape[B,h,Nk,d_v]

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # shape[B,h,Nq,Nk]
        att = torch.softmax(att, -1) # shape[B,h,Nq]
        # att = torch.softmax(att, -1) # shape[B,h,Nq]
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, Nq, self.h * self.d_v)  # shape[B,Nq,h*d_v]
        out = self.fc_o(out).permute(0, 2, 1)  # shape[B,d_out,Nq]
        # whether to use skip-connect Path?
        return out

class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', coord=True, atten=False, out_dim=2):
        super(Snake, self).__init__()
        # feature_dim: the input channels
        # state_dim: the mid channels for snake
        self.coord = coord
        self.atten = atten
        feature_dim = feature_dim + 1 if coord else feature_dim
        self.head = BasicBlock(feature_dim, state_dim, conv_type) #feature_dim is the input_channels, state_dim is the output_channels
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i])
            self.__setattr__('res'+str(i), conv)

        atten_layer_num = 0
        if atten:
            d_queries = state_dim
            # self.atten_net_coord = Snake_Attention_coord(d_queries,h=3)
            self.atten_net = Snake_Attention(d_queries,h=3)
            atten_layer_num = 1

        fusion_state_dim = 256
        cat_dim = state_dim * (self.res_layer_num + 1 + atten_layer_num)
        self.fusion = nn.Conv1d(cat_dim, fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(cat_dim + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim, 1)
        )

    def coord_conv(self, feat):
        with torch.no_grad():
            L = feat.shape[-1]
            pos1 = torch.linspace(0, 1, L//2, device=feat.device)
            pos2 = torch.linspace(1, 0, L-L//2, device=feat.device)
            pos = torch.cat([pos1,pos2],dim=0).unsqueeze(0).unsqueeze(0).expand(feat.shape[0], 1, -1)
        feat = torch.cat([feat, pos], dim=1)
        return feat

    def forward(self, x, adj):
        if self.coord:
            x = self.coord_conv(x)
        states = []

        x = self.head(x, adj)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, adj) + x
            states.append(x)
        if self.atten:
            atten_state = self.atten_net(x)
            states.append(atten_state)
        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x

class Conv1x_Upsample_Block(nn.Module):
    def __init__(self,schedule_up=[16,64,128],in_channels=64):
        super(Conv1x_Upsample_Block,self).__init__()
        self.schedule_up = schedule_up
        self.num_convs = len(schedule_up)
        kernel_list = [7,7,7]
        in_channels = in_channels
        activation = nn.ReLU
        self.up_block=nn.Sequential()
        for i in range(self.num_convs):
            conv = nn.Conv1d(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_list[i],
                             padding=3) # 'same' padding
            self.up_block.add_module(f'conv_{i}',conv)
            activate = activation(inplace=True)
            self.up_block.add_module(f'activate_{i}',activate)

    def forward(self,x):
        # B,Ninst,Cin = x.shape
        # x = x.reshape(B*Ninst,Cin,1)
        Lout = self.schedule_up[-1]
        x = x.repeat(1,1,Lout)
        x = self.up_block(x)
        return x





