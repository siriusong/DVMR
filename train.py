import argparse
import copy
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from util.data_newlabel_undirect_sim import *
from model.DVR import *
from torch.autograd import Variable
from util.misc import CSVLogger
from sklearn.metrics import classification_report,precision_recall_curve,roc_auc_score,auc,average_precision_score
from torch_geometric.data import Data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=60,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--in_dim',
                    type=int,
                    default=47 + 657 ,
                    help='dim of atom feature')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--seed',
                    type=int,
                    default=123,
                    help='random seed (default: 123)')
parser.add_argument('--logdir', type=str, default='noadsclogs', help='logdir name')
parser.add_argument('--dataset', type=str, default='USPTO50K', help='dataset')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
parser.add_argument('--heads', type=int, default=4, help='number of heads')
parser.add_argument('--gat_layers',
                    type=int,
                    default=3,
                    help='number of gat layers')
parser.add_argument('--valid_only',
                    action='store_true',
                    default=False,
                    help='valid_only')
parser.add_argument('--test_only',
                    action='store_true',
                    default=False,
                    help='test_only')
parser.add_argument('--test_on_train',
                    action='store_true',
                    default=False,
                    help='run testing on training data')
parser.add_argument('--typed',
                    action='store_true',
                    default=False,
                    help='if given reaction types')
parser.add_argument('--use_cpu',
                    action='store_true',
                    default=False,
                    help='use gpu or cpu')
parser.add_argument('--load',
                    action='store_true',
                    default=False,
                    help='load model checkpoint.')

args = parser.parse_args()


def collate(data):
    return map(list, zip(*data))


def test(GAT_model, test_dataloader, data_split='valid', save_pred=False, bestacc=0,files=None,bondfeatures= None):
    GAT_model.eval()
    correct = 0.
    total = 0.
    epoch_loss = 0.
    dis_num_correct = 0.
    true_bond_label = []
    pre_bond_label = []
    true_nums = []
    pre_nums = []

    # Bond disconnection probability
    pred_true_list = []
    pred_logits_mol_list = []
    # Bond disconnection number gt and prediction
    bond_change_gt_list = []
    bond_change_pred_list = []
    true_list = []
    pred_score_list = []
    for i, data in enumerate(tqdm(test_dataloader)):
        rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num = data

        x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
        x_pattern_feat = list(
            map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
        x_atom = list(
            map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                x_pattern_feat))

        if args.typed:
            rxn_class = list(
                map(lambda x: torch.from_numpy(x).float(), rxn_class))
            x_atom = list(
                map(lambda x, y: torch.cat([x, y], dim=1), x_atom, rxn_class))

        x_atom = torch.cat(x_atom, dim=0)
        disconnection_num = torch.LongTensor(disconnection_num)
        if not args.use_cpu:
            x_atom = x_atom.cuda()
            disconnection_num = disconnection_num.cuda()
            bondfeatures = [i.cuda() for i in bondfeatures]
        pygraph = [Data(edge_index=torch.stack(graph.adj_sparse('coo')).long().cuda()) for graph in x_graph]

        x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
        y_adj = list(map(lambda x: torch.from_numpy(np.array(x)), y_adj))
        if not args.use_cpu:
            x_adj = [xa.cuda() for xa in x_adj]
            y_adj = [ye.cuda() for ye in y_adj]

        mask = list(map(lambda x: x.contiguous().view(-1, 1).bool(), x_adj))
        bond_disconnections = list(
            map(lambda x, y: torch.masked_select(x.contiguous().view(-1, 1), y), y_adj,
                mask))
        bond_labels = torch.cat(bond_disconnections, dim=0).float()
        g_dgl = dgl.batch(x_graph)

        h_readout1,eh1,h_pred, e_pred = GAT_model(g_dgl, x_atom,bondfeatures,pygraph)
        e_pred = e_pred.squeeze()
        loss_h = nn.CrossEntropyLoss(reduction='sum')(h_pred,
                                                      disconnection_num)
        loss_ce = nn.BCEWithLogitsLoss(reduction='sum')(e_pred, bond_labels)
        loss = loss_ce + loss_h
        epoch_loss += loss.item()

        pred_score = torch.sigmoid(e_pred).cpu().detach().numpy()
        pred_true = bond_labels.tolist()
        ones = np.ones_like(pred_score, dtype=float)
        true_list.extend(pred_true)
        pred_score_list.extend(pred_score)


        h_pred = torch.argmax(h_pred, dim=1)
        pre_nums.extend(h_pred.tolist())
        true_nums.extend(disconnection_num.tolist())
        true_bond_label.extend(bond_labels.tolist())
        bond_change_pred_list.extend(h_pred.cpu().tolist())
        bond_change_gt_list.extend(disconnection_num.cpu().tolist())

        start = end = 0
        pred = torch.sigmoid(e_pred)
        edge_lens = list(map(lambda x: x.shape[0], bond_disconnections))
        cur_batch_size = len(edge_lens)
        bond_labels = bond_labels.long()
        for i in range(len(disconnection_num)):
            if torch.equal(disconnection_num[i],h_pred[i]):
                dis_num_correct += 1
        for j in range(cur_batch_size):
            start = end
            end += edge_lens[j]
            label_mol = bond_labels[start:end]
            pred_proab = pred[start:end]
            mask_pos = torch.nonzero(x_adj[j]).tolist()
            assert len(mask_pos) == len(pred_proab)

            pred_disconnection_adj = torch.zeros_like(x_adj[j], dtype=torch.float32)
            for idx, pos in enumerate(mask_pos):
                pred_disconnection_adj[pos[0], pos[1]] = pred_proab[idx]
            for idx, pos in enumerate(mask_pos):
                pred_proab[idx] = (pred_disconnection_adj[pos[0], pos[1]] + pred_disconnection_adj[pos[1], pos[0]]) / 2

            pred_mol = pred_proab.round().long()
            pre_bond_label.extend(pred_mol.tolist())

            if torch.equal(pred_mol, label_mol):
                correct += 1
                pred_true_list.append(True)
                pred_logits_mol_list.append([
                    True,
                    label_mol.tolist(),
                    pred_proab.tolist(),
                ])
            else:
                pred_true_list.append(False)
                pred_logits_mol_list.append([
                    False,
                    label_mol.tolist(),
                    pred_proab.tolist(),
                ])
            total += 1

    pred_lens_true_list = list(
        map(lambda x, y: x == y, bond_change_gt_list, bond_change_pred_list))
    bond_change_pred_list = list(
        map(lambda x, y: [x, y], bond_change_gt_list, bond_change_pred_list))
    if save_pred:
        print('pred_true_list size:', len(pred_true_list))
        np.savetxt('noadsclogs/{}_disconnection_{}.txt'.format(data_split, args.exp_name),
                   np.asarray(bond_change_pred_list),
                   fmt='%d')
        np.savetxt('noadsclogs/{}_result_{}.txt'.format(data_split, args.exp_name),
                   np.asarray(pred_true_list),
                   fmt='%d')
        with open('noadsclogs/{}_result_mol_{}.txt'.format(data_split, args.exp_name),
                  'w') as f:
            for idx, line in enumerate(pred_logits_mol_list):
                f.write('{} {}\n'.format(idx, line[0]))
                f.write(' '.join([str(i) for i in line[1]]) + '\n')
                f.write(' '.join([str(i) for i in line[2]]) + '\n')

    print('Bond disconnection number prediction acc: {:.6f}'.format(
        np.mean(pred_lens_true_list)))


    print('Loss: ', epoch_loss / total)
    acc = correct / total
    dis_num_acc = dis_num_correct/total
    print("before best_acc is:{} ".format(bestacc))
    if bestacc < acc and bestacc:

        torch.save(GAT_model.state_dict(),
                   'checkpoints_3/{}best_checkpoint.pt'.format(args.exp_name))
    print('Bond disconnection acc (without auxiliary task): {:.6f}'.format(acc))
    print('Bond disconnection acc (without auxiliary task): {:.6f}'.format(dis_num_acc))

    sk_report = classification_report(true_bond_label, pre_bond_label,digits=6)
    files.write("\n" + data_split + " bond result" + "\n")
    files.write(sk_report)
    files.flush()
    sk_report = classification_report(true_nums,pre_nums,digits=6 )
    files.write("dis num  result" + "\n")
    files.write(sk_report)
    files.flush()
    pred_score = np.subtract(ones, pred_score)
    ap = average_precision_score(pred_true, pred_score, pos_label=0, average=None)
    #
    ones = np.ones_like(pred_true, dtype=float)
    pred_true = np.subtract(ones, pred_true)
    auc_ = roc_auc_score(pred_true, pred_score)
    files.write("bond_label auc:" + "\n")
    files.write(str(auc_) + "\n")
    files.write("bond_label ap:" + "\n")
    files.write(str(ap))
    files.flush()
    return acc, dis_num_acc

def gen_ran_output( model, vice_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for (adv_name,vice_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if len(name.split('.')) >5:
            if name.split('.')[4] == 'aggre_embed_edge'or name.split('.')[4] == 'node_sf_attten':
                vice_param.data = param.data
            else:
                ones = torch.ones_like(param.data)
                data_std = param.data.std(unbiased=False)
                data = ones*data_std
                vice_param.data = param.data + 0.4* torch.normal(0,torch.abs(data)).to(device)
        else:
            vice_param.data = param.data

    return vice_model
if __name__ == '__main__':
    typed = True
    first = False
    for numlayer in [3,4,5]:
        local_acc = 0
        batch_size = 128
        epochs = args.epochs
        data_root = os.path.join('data', args.dataset)
        args.exp_name = args.dataset
        if args.typed and first == False:
            args.in_dim += 10
            args.exp_name += '_typed'
            typed = True
            first = True
        elif first == True:
            typed = True
            args.exp_name += '_typed'

        else:
            typed = False

            args.exp_name += '_untyped'
        args.exp_name += "_egat_dvml_nonumber_{}".format(numlayer)

        print(args)
        test_id = '{}'.format(args.logdir)
        filename = 'noadsclogs/' + test_id + args.exp_name + '.csv'
        sk_filename = 'new_sclogs/' + test_id + args.exp_name + '.txt'
        file = open(sk_filename, 'a')
        train_data = RetroCenterDatasets(root=data_root, data_split='train')
        bond_features = train_data.get_bond_feats2()
        GAT_model = GATNet(
            in_dim=args.in_dim,
            num_layers=numlayer,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            use_gpu=(args.use_cpu == False),
            bond_feature = bond_features[1].size(1)

        )

        if args.use_cpu:
            device = 'cpu'
        else:
            GAT_model = GAT_model.cuda()

            device = 'cuda:0'

        if args.load:

            GAT_model.load_state_dict(
                torch.load('checkpoints_3/USPTO50K_untypedgatv2_gai12_noAD_{}best_checkpoint.pt'.format(numlayer),
                           map_location=torch.device(device)), )
            args.lr *= 0.2
            milestones = []
        else:
            milestones = [20,40]
        # if numlayer >= 3:
        #     lr = 1e-4
        # else:
        #     lr = args.lr
        lr = args.lr
        optimizer = torch.optim.Adam([{
            'params': GAT_model.parameters()
        }],
            lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

        if args.test_only:
            test_data = RetroCenterDatasets(root=data_root, data_split='test')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1 * batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=collate)
            test(GAT_model, test_dataloader, data_split='test', save_pred=True,files=file,bondfeatures=test_data.get_bond_feats2())
            exit(0)

        valid_data = RetroCenterDatasets(root=data_root, data_split='valid')
        valid_dataloader = DataLoader(valid_data,
                                      batch_size=1 * batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=collate)
        if args.valid_only:
            test(GAT_model, valid_dataloader)
            exit(0)




        train_dataloader = DataLoader(train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=collate)
        if args.test_on_train:
            bond_features = train_data.get_bond_feats2()

            test_train_dataloader = DataLoader(train_data,
                                               batch_size=1 * batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               collate_fn=collate)
            test(GAT_model, test_train_dataloader, data_split='train', save_pred=True,bondfeatures=bond_features)
            exit(0)
        csv_logger = CSVLogger(
            args=args,
            fieldnames=['epoch', 'train_acc',
                        'valid_acc','valid_num_acc',
                        'train_loss'],
            filename=filename,
        )

        # Record epoch start time
        for epoch in range(1, 1 + epochs):
            total = 0.
            correct = 0.

            epoch_loss = 0.
            epoch_loss_ce = 0.
            pre_bond_label = []
            true_bond_label = []
            pre_nums = []
            true_nums = []
            atom_correct = 0.
            epoch_loss_h = 0.
            GAT_model.train()


            progress_bar = tqdm(train_dataloader,ncols=200)
            for i, data in enumerate(progress_bar):
                progress_bar.set_description('Epoch ' + str(epoch))
                rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num = data

                x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
                x_pattern_feat = list(
                    map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
                x_atom = list(
                    map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                        x_pattern_feat))
                # x_groups=list(map(lambda x: torch.from_numpy(x).float(), x_groups))
                # x_groups=torch.cat(x_groups,dim=1)
                if typed:
                    rxn_class = list(
                        map(lambda x: torch.from_numpy(x).float(), rxn_class))
                    x_atom = list(
                        map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                            rxn_class))

                x_atom = torch.cat(x_atom, dim=0)
                disconnection_num = torch.LongTensor(disconnection_num)
                if not args.use_cpu:
                    x_atom = x_atom.cuda()
                    disconnection_num = disconnection_num.cuda()
                    bond_features = [i.cuda() for i in bond_features]
                pygraph = [Data(edge_index=torch.stack(graph.adj_sparse('coo')).long().cuda(),num_nodes=graph.num_nodes()) for graph in x_graph]
                x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
                y_adj = list(map(lambda x: torch.from_numpy(np.array(x)), y_adj))
                if not args.use_cpu:
                    x_adj = [xa.cuda() for xa in x_adj]
                    y_adj = [ye.cuda() for ye in y_adj]

                mask = list(map(lambda x: x.contiguous().view(-1, 1).bool(), x_adj))  
                bond_connections = list(
                    map(lambda x, y: torch.masked_select(x.contiguous().view(-1, 1), y), y_adj,
                        mask))  
                bond_labels = torch.cat(bond_connections, dim=0).float()

                g_dgl = dgl.batch(x_graph)

                GAT_model.zero_grad()
                # # batch graph
                rand_model = copy.deepcopy(GAT_model)

                rand_model = gen_ran_output(GAT_model, rand_model)
                loss_ce_cl = 0
                loss_h_cl = 0
                # # # batch graph
                h_readout1,eh1,h_pred, e_pred = GAT_model(g_dgl, x_atom, bond_features,pygraph)
                g_dgl.edata.pop('w')
                h_readout2,eh2= rand_model(g_dgl, x_atom, bond_features,pygraph,pred=False)
                eh2 = Variable(eh2.detach().data, requires_grad=False)
                h_readout2 = Variable(h_readout2.detach().data, requires_grad=False)
                loss_h_cl = GAT_model.loss_cl(h_readout1,h_readout2)
                loss_ce_cl = GAT_model.loss_cl(eh1,eh2)

                e_pred = e_pred.squeeze()
                loss_h = nn.CrossEntropyLoss(reduction='sum')(h_pred,
                                                              disconnection_num)
                loss_ce = nn.BCEWithLogitsLoss(reduction='sum')(e_pred,
                                                                bond_labels)
                loss = loss_ce +loss_h_cl+loss_ce_cl
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_ce += loss_ce.item()
                epoch_loss_h += loss_h.item()
                pre_num = torch.argmax(h_pred,dim=1)
                start = end = 0
                pred = torch.round(torch.sigmoid(e_pred)).long()
                true_bond_label.extend(bond_labels.tolist())
                pre_bond_label.extend(pred.tolist())
                pre_nums.extend(pre_num.tolist())
                true_nums.extend(disconnection_num.tolist())
                edge_lens = list(map(lambda x: x.shape[0], bond_connections))
                cur_batch_size = len(edge_lens)
                bond_labels = bond_labels.long()

                for i in range(len(disconnection_num)):
                    if torch.equal(pre_num[i],disconnection_num[i]):
                        atom_correct += 1
                for j in range(cur_batch_size):
                    start = end
                    end += edge_lens[j]
                    if torch.equal(pred[start:end], bond_labels[start:end]):
                        correct += 1
                assert end == len(pred)

                total += cur_batch_size
                progress_bar.set_postfix(
                    loss='%.5f' % (epoch_loss / total),
                    numacc='%.5f' % (atom_correct / total),

                    acc='%.5f' % (correct / total),
                    loss_ce='%.5f' % (epoch_loss_ce / total),
                    loss_h='%.5f' % (epoch_loss_h / total),
                )
            scheduler.step()

            train_acc = correct / total
            num_acc = atom_correct/total
            train_loss = epoch_loss / total
            print('Train Loss: {:.5f}'.format(train_loss))
            print('Train Bond Disconnection Acc: {:.5f}'.format(train_acc))
            print('Train  Disconnection num Acc: {:.5f}'.format(num_acc))

            sk_report = classification_report(true_bond_label, pre_bond_label)
            file.write("\n"   + "train bond result" + "\n")
            file.write(sk_report)
            file.flush()
            sk_report = classification_report(true_nums, pre_nums)
            file.write("dis num result" + "\n")
            file.write(sk_report)
            file.flush()
            if epoch % 5 == 0:

                valid_acc,valid_num_acc= test(GAT_model, valid_dataloader, bestacc=local_acc,files=file,bondfeatures=bond_features)
                if valid_acc > local_acc:
                    local_acc = valid_acc
                row = {
                    'epoch': str(epoch),
                    'train_acc': str(train_acc),

                    'valid_acc': str(valid_acc),
                    'valid_num_acc':str(valid_num_acc),
                    'train_loss': str(train_loss),
                }
                csv_logger.writerow(row)

        torch.save(GAT_model.state_dict(),
                   'checkpoints_3/{}_checkpoint.pt'.format(args.exp_name))
        GAT_model.load_state_dict(
            torch.load('checkpoints_3/{}best_checkpoint.pt'.format(args.exp_name),
                       map_location=torch.device(device)), )
        test_data = RetroCenterDatasets(root=data_root, data_split='test')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1 * batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=collate)
        acc,test_num_acc = test(GAT_model, test_dataloader, data_split='test', save_pred=True,files=file,bondfeatures=bond_features)
        row = {
            'epoch': 'test_result',
            'train_acc': str(0),
            'valid_acc': str(acc),
            'valid_num_acc': str(test_num_acc),
            'train_loss': str(0),
        }
        csv_logger.writerow(row)

        csv_logger.close()
