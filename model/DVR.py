import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from dgl.nn.pytorch import GATv2Conv,EGATConv


class interGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_in_dim,e_out_dim,bond_feature):
        super(interGATLayer, self).__init__()
        self.embed_node = nn.Linear(in_dim, out_dim, bias=False)
        # self.node_sf_attten=nn.Linear(out_dim*2,1)
        self.attn_fc = nn.Linear(out_dim , 1, bias=False)
        self.inter_fuse = nn.Linear(out_dim*2+e_in_dim,out_dim)

        self.to_node_fc = nn.Linear(out_dim+e_in_dim+out_dim, out_dim, bias=False)
        self.edge_nor=nn.BatchNorm1d(num_features=e_in_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.e_in_dim=e_in_dim
        self.aggre_embed_edge=nn.Linear(out_dim*2+2*e_in_dim,e_out_dim,bias=False)
        self.edge_drop=nn.Dropout(p=0.1)
        self.node_drop=nn.Dropout(p=0.1)
        self.attention_drop=nn.Dropout(p=0.3)
        self.fc_self = nn.Linear(in_dim, out_dim, bias=False)
        self.embed_edge = nn.Linear(e_in_dim, e_in_dim, bias=False)
        self.edge_sf_atten=nn.Linear(e_in_dim,1,bias=False)
        self.edge_linear=nn.Linear(e_in_dim*3,e_in_dim)
        self.concentrate_h = nn.Linear(out_dim*2+e_in_dim,out_dim)
        self.en_score = nn.Linear(32+e_in_dim,1)
        self.att_sat = nn.Linear(out_dim*2+e_in_dim, 1)
        self.en_in = e_in_dim
        self.out_channels = out_dim
        self.w_score = nn.Linear(64*2,1)
        self.en_embedding = nn.Sequential(
            nn.Linear(bond_feature, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.Linear(32, e_in_dim),
            nn.BatchNorm1d(e_in_dim),
            nn.ELU(inplace=True)
        )
        self.en_embedding.requires_grad_(False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.embed_node.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.embed_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
        nn.init.xavier_normal_(self.aggre_embed_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.to_node_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_linear.weight, gain=gain)
        nn.init.xavier_normal_(self.concentrate_h.weight, gain=gain)
        nn.init.xavier_normal_(self.en_score.weight, gain=gain)
        nn.init.xavier_normal_(self.att_sat.weight, gain=gain)
        nn.init.xavier_normal_(self.w_score.weight, gain=gain)
        nn.init.xavier_normal_(self.inter_fuse.weight,gain=gain)

    def enviro_cal(self,g,bond_feats):
        if 'w' in g.edata.keys():
            return
        new_bond_feats_list = []
        features = []
        for index in range(len(bond_feats)):
            if index == 0:
                if torch.cuda.is_available():
                    zeros = torch.from_numpy(np.zeros(self.en_in,dtype=np.float32)).cuda()
                else:
                    zeros = torch.from_numpy(np.zeros(self.en_in, dtype=np.float32))
                new_bond_feats_list.append(zeros)
                continue
            new_bond_feats_list.append(torch.mean((self.en_embedding(bond_feats[index])),dim=0))
        bond_type = g.edata['type'].cpu().numpy()
        bond_types = bond_type[:, 0]
        for index, i in enumerate(bond_types):
            features.append(new_bond_feats_list[int(bond_types[index])])
        features = torch.stack(features)
        g.edata['w'] = features

    def self_attention(self,edges):
        edge_embed=self.embed_edge(edges.data['w'])
        edges.data['ow'] = edge_embed
        edge_self_attention=self.edge_sf_atten(F.leaky_relu(edge_embed,negative_slope=0.1))
        edge_embed=edge_self_attention*edge_embed
        return {'w':edge_embed}
    
    def inter_attention(self, edges):
        sat = torch.cat([edges.src['h'], edges.dst['h'],edges.data['w']],
                       dim=1)
        z2 = self.inter_fuse(sat)
        edge_weight=F.leaky_relu(z2, negative_slope=0.1)
        edge_weight = self.attn_fc(edge_weight)
        edge_weight = self.attention_drop(edge_weight)
        edge_embed=edge_weight*edges.data['w']
        return {'inw': F.leaky_relu(z2, negative_slope=0.1), 'w': edge_embed,'sat':sat}

    def message_func2(self,edges):
        return {'sh':edges.src['h'],'dh':edges.dst['h'],'w':edges.data['w'],'inw':edges.data['inw'],'h': edges.src['h'],
            'w':edges.data['w']}

    def reduce_func2(self,nodes):
        attention = self.attn_fc(nodes.mailbox['inw'])
        alpha =F.softmax(attention, dim=1)
        t=torch.cat([nodes.mailbox['sh'], nodes.mailbox['w']],dim=-1)
        h = alpha * t
        h = torch.sum(h,dim=1)
        t = torch.cat([h, nodes.data['oh']] ,dim=-1)
        h = self.concentrate_h(t)
        h=self.node_drop(h)
        return {'h': h,}

    def edge_calc(self, edges):
        w = self.edge_nor(edges.data['w'])
        ow = edges.data['ow']
        z2 = torch.cat([edges.src['h'], edges.dst['h'],w,ow],
                      dim=1)
        w = self.aggre_embed_edge(z2)
        w=self.edge_drop(w)
        return {'w': w}
    def forward(self, g, h,bond_feats,use_env=False):
        with g.local_scope():
            g.ndata['h'] = self.embed_node(h)
            g.ndata['oh']=g.ndata['h']
            if use_env:
                self.enviro_cal(g,bond_feats)
            g.edata['ow']=g.edata['w']
            g.apply_edges(self.self_attention)
            g.apply_edges(self.inter_attention)
            g.update_all(self.message_func2, self.reduce_func2)
            g.apply_edges(self.edge_calc)
            return g.ndata['h'], g.edata['w'],g.edata['sat']

class MultiHeadGATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 e_in_dim,
                 e_out_dim,
                 num_heads,
                 bond_feature,
                 use_gpu=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList().to('cuda')
        self.use_gpu = use_gpu
        for i in range(num_heads):
            self.heads.append(interGATLayer(in_dim, out_dim, e_in_dim, e_out_dim,bond_feature))

    def forward(self, g, h,bond_feats,merge,use_env=False):

        if self.use_gpu:
            g=g.to(torch.device('cuda'))
        outs = list(map(lambda x: x(g, h,bond_feats,use_env=use_env), self.heads))
        outs = list(map(list, zip(*outs)))
        head_outs = outs[0]
        edge_outs = outs[1]
        sat = outs[2]
        if merge == 'flatten':
            head_outs = torch.cat(head_outs, dim=1)
            edge_outs = torch.cat(edge_outs, dim=1)
        elif merge == 'mean':
            head_outs = torch.mean(torch.stack(head_outs), dim=0)
            edge_outs = torch.mean(torch.stack(edge_outs), dim=0)
        g.edata['w'] = edge_outs
        sat = torch.mean(torch.stack(sat),dim=0)
        return head_outs, edge_outs,sat



class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads,bond_feature, use_gpu=True):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.gat = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.drop = nn.Dropout(0.2)
        self.hidden = hidden_dim
        self.e_same_dim = nn.Linear(hidden_dim*heads,hidden_dim)
        self.e_score = nn.Linear(hidden_dim*2,1)
        self.e_enb = nn.Linear(hidden_dim*self.num_layers*2,hidden_dim)
        self.gru_1 = nn.GRU(input_size=hidden_dim,hidden_size=hidden_dim).cuda()
        self.heads = heads
        self.gru_2 = nn.GRU(input_size=hidden_dim*heads,hidden_size=hidden_dim*heads).cuda()
        self.gatv2 = nn.ModuleList()
        #GIN
        # mlp = nn.Sequential(
        #     nn.Linear(in_dim,hidden_dim),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Linear(hidden_dim,hidden_dim),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        # )
        # mlp2 = nn.Sequential(
        #     nn.Linear(hidden_dim*heads,hidden_dim*heads),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_dim*heads),
        #     nn.Linear(hidden_dim*heads,hidden_dim*heads),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_dim*heads),
        # )
        #abl
        # mlp = nn.Sequential(
        #     nn.Linear(in_dim,hidden_dim),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim,hidden_dim),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim,hidden_dim),
        #
        # )
        # mlp2 = nn.Sequential(
        #     nn.Linear(hidden_dim*heads,hidden_dim*heads),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim*heads,hidden_dim*heads),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim*heads,hidden_dim*heads),
        #
        # )
        self.gcn.append(
            EGATConv(in_node_feats=in_dim,
                          in_edge_feats=12,
                          out_node_feats=hidden_dim,
                          out_edge_feats=hidden_dim,
                          num_heads=heads)
            # GATv2Conv(in_dim,hidden_dim,heads)
            # GINEConv(mlp)
            # GINConv(mlp)
            # mlp
            # GATConv(in_dim,hidden_dim,4)
            # GraphConv(in_dim,hidden_dim)
        )
        self.gat.append(
            # MultiHeadGATLayer(hidden_dim, hidden_dim, 12,hidden_dim, heads,bond_feature, use_gpu,))

            # attenion need mul heads
            MultiHeadGATLayer(hidden_dim*self.heads, hidden_dim, 12,hidden_dim, heads,bond_feature, use_gpu,))
            # MultiHeadGATLayer(hidden_dim, hidden_dim, 13,hidden_dim, heads, use_gpu))
        self.gatv2.append(GATv2Conv(hidden_dim*2+12,hidden_dim,heads))
        for l in range(1, num_layers):
            self.gat.append(
                MultiHeadGATLayer(
                    hidden_dim*heads,
                    hidden_dim,
                    hidden_dim*self.heads,
                    hidden_dim,
                    heads,
                    bond_feature,
                    use_gpu,

                ))
            # self.gcn.append(GINConv(mlp2))
            # self.gcn.append(mlp2)

            # self.gcn.append(GATv2Conv(hidden_dim*self.heads,hidden_dim,heads))
            # self.gcn.append(GraphConv(hidden_dim*self.heads,hidden_dim*self.heads))
            self.gcn.append(EGATConv(in_node_feats=hidden_dim*self.heads,
                          in_edge_feats=hidden_dim*self.heads,
                          out_node_feats=hidden_dim,
                          out_edge_feats=hidden_dim,
                          num_heads=heads))

            self.gatv2.append(GATv2Conv(hidden_dim*2+hidden_dim*self.heads,hidden_dim,heads))
        self.linear_e = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        #EGAT
        self.en_embedding = nn.Sequential(
            nn.Linear(112, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.Linear(32, 12),
            nn.BatchNorm1d(12),
            nn.ELU(inplace=True)
        )
        self.en_embedding.requires_grad_(False)
        self.linear_h = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        loss = 1/T - torch.log(sim_matrix).mean()
        return loss
    
    def forward(self, g:dgl.DGLGraph, h,bond_feats,pygraph,pred=True):

        new_g = dgl.line_graph(g)
        new_g = dgl.add_self_loop(new_g)
        if torch.cuda.is_available():
            g=g.to(torch.device('cuda'))
            h = h.cuda()
            new_g = new_g.to(torch.device('cuda'))
        # EGAT
        if 'w' not in g.edata.keys():
            new_bond_feats_list = []
            features = []
            for index in range(len(bond_feats)):
                if index == 0:
                    if torch.cuda.is_available():
                        zeros = torch.from_numpy(np.zeros(12, dtype=np.float32)).cuda()
                    else:
                        zeros = torch.from_numpy(np.zeros(12, dtype=np.float32))
                    new_bond_feats_list.append(zeros)
                    continue
                new_bond_feats_list.append(torch.mean((self.en_embedding(bond_feats[index])), dim=0))
            bond_type = g.edata['type'].cpu().numpy()
            bond_types = bond_type[:, 0]
            for index, i in enumerate(bond_types):
                features.append(new_bond_feats_list[int(bond_types[index])])
            features = torch.stack(features)
            g.edata['w'] = features
        init_data = torch.zeros([1, g.num_edges(), self.hidden*self.heads],device=h.device)
        total_e = None
        total_sat = None

        for sss in range(self.num_layers-1):
            # EGAT
            h,_ = self.gcn[sss](g,h,g.edata['w'])
            sh = torch.flatten(h,start_dim=1,end_dim=2)
            h, e,sat = self.gat[sss](g, sh,bond_feats, merge='flatten',use_env=True)
            sat = self.gatv2[sss](new_g,sat)
            sat = torch.mean(sat,dim=1)
            sat = F.elu(sat)
            eh = e.unsqueeze(0)
            for i in range(self.num_layers):
                eh, init_data = self.gru_2(eh, init_data)
                eh = self.drop(eh)
                init_data = self.drop(init_data)
            same_e = self.e_same_dim(e)
            lo_e = F.leaky_relu(torch.cat([same_e,sat],dim=1),negative_slope=0.1)
            score = self.e_score(lo_e)

            if total_e == None:
                total_e = score*torch.cat([same_e,sat],dim=1)

            else:
                total_e = torch.cat([total_e,score*torch.cat([same_e,sat],dim=1)],dim=1)
            e = eh.squeeze()
            h = F.elu(h)
            e = F.elu(e)
            g.edata['w']=e

        # EGAT
        h,_ = self.gcn[-1](g,h,g.edata['w'])
        h = torch.flatten(h, start_dim=1, end_dim=2)
        h, e ,sat = self.gat[-1](g, h,bond_feats, merge='mean',use_env=True)
        sat = self.gatv2[-1](new_g, sat)
        sat = torch.mean(sat, dim=1)


        batch_size = e.size(0)
        eh = e.unsqueeze(0)
        init_data = torch.zeros([1,batch_size,128],device=h.device)
        for i in range(self.num_layers):
            eh,init_data = self.gru_1(eh,init_data)
            eh = self.drop(eh)
            init_data = self.drop(init_data)
        e = eh.squeeze()
        lo_e = F.leaky_relu(torch.cat([e, sat], dim=1), negative_slope=0.1)

        score = self.e_score(lo_e)
        if total_e == None:
            total_e = score * torch.cat([e, sat], dim=1)

        else:
            total_e = torch.cat([total_e, score * torch.cat([e, sat], dim=1)], dim=1)
        e = self.e_enb(total_e)
        g.edata['h'] = e
        g.ndata['h'] = h


        g.edata['w'] = dgl.softmax_edges(g, 'h')
        e_readout = dgl.readout_edges(g, 'h', 'w')
        h_readout = dgl.mean_nodes(g, 'h')
        eh = torch.cat((h_readout,e_readout),1)

        if pred:

            h_pred = self.linear_h(h_readout)
            # # Edge prediction
            eh = dgl.broadcast_edges(g, eh)
            e_fused = torch.cat((eh, e), dim=1)
            e_pred = self.linear_e(e_fused)

            return h_readout,e_readout, h_pred,e_pred
        else:
            # g.edata.pop('w')
            # model = self.random_params()
            # for l in range(self.num_layers - 1):
            #     h2, e2 = model[l](g, old_h, bond_feats, merge='flatten', use_env=True)
            #     old_h = F.elu(h2)
            #     g.edata['w'] = e2
            #
            # _, e2 = model[-1](g, h2, bond_feats, merge='mean', use_env=True)
            # e2 = Variable(e2.detach().data, requires_grad=False)
            # g.edata['h1'] = e
            # g.edata['h2'] = e2
            # g.edata['w1'] = dgl.softmax_edges(g, 'h1')
            # g.edata['w2'] = dgl.softmax_edges(g, 'h2')
            # e_readout1 = dgl.readout_edges(g, 'h1', 'w1')
            # e_readout2 = dgl.readout_edges(g, 'h2', 'w2')
            #
            # # g.ndata['h1'] = h
            # # g.ndata['h2'] = h2
            # # h_readout_1 = dgl.mean_nodes(g,'h1')
            # # h_readout_2 = dgl.mean_nodes(g,'h2')
            #
            # loss_e = self.loss_cl(e_readout1,e_readout2)
            # loss_h = 0
            return h_readout,e_readout