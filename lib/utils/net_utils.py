import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from termcolor import colored
import glob
import os.path as osp



def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y
def tanh(x):
    y = torch.clamp(x.tanh(), min=-1+1e-4, max=1 - 1e-4)
    return y

def _neg_loss(pred, gt,th=1):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    # pos_inds = gt.eq(1).float()
    pos_inds = gt.ge(th).float() # th should be in [0,1]
    neg_inds = gt.lt(th).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target,th=1.0):
        return self.neg_loss(out, target,th)


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, normalize=True, reduce=True):
    """
    :param vertex_pred:     [b, vn*2, h, w]
    :param vertex_targets:  [b, vn*2, h, w]
    :param vertex_weights:  [b, 1, h, w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss





class HausdorffLoss(nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()
        self.pnum=128

    def unifrom_resample(self,poly, newpnum):
        pnum, cnum = poly.shape
        assert cnum == 2
        idxnext = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        polynext = poly[idxnext]  # shift edge to one index
        edgelen = torch.sqrt(torch.sum((polynext - poly) ** 2, dim=1))  # calc the edge len
        edgeidxsort = torch.argsort(edgelen)
        edgelen_cumsum = torch.cumsum(edgelen,dim=0)
        edgelentotal = edgelen.sum()
        edgelennew = edgelentotal/newpnum
        # find the begin and end point
        device = poly.device
        idx_npnew = torch.arange(newpnum).cuda(device)
        target_edgelen = edgelennew*idx_npnew.type(torch.float)
        target_edgelen_mat = target_edgelen.repeat((pnum,1)).transpose(1,0)
        diff = edgelen_cumsum.repeat((newpnum,1)) - target_edgelen_mat

        idx_np = torch.flip(torch.arange(1,pnum+1),dims=[0]).cuda(device)
        diff_mask = (diff>=0).type_as(idx_np) # find first one
        diff_idx = diff_mask * idx_np.repeat((newpnum,1))
        idxb_npnew = torch.argmax(diff_idx,dim=1) #[Np2,1]
        idxe_npnew = (idxb_npnew+1)%pnum
        # do interpolation
        init_zero = edgelen_cumsum[0:1]*0.0
        edgelen_cumsum = torch.cat((init_zero,edgelen_cumsum),0) # Np2+1
        pb_npnewx2 = torch.index_select(poly,0,idxb_npnew)
        pe_npnewx2 = torch.index_select(poly,0,idxe_npnew)
        taredgelen_npnew = target_edgelen
        edgelencumsum_npnew = torch.gather(edgelen_cumsum,0,idxb_npnew)
        edgelen_npnew = torch.gather(edgelen,0,idxb_npnew)
        wnp_npnewx1 = (taredgelen_npnew - edgelencumsum_npnew)/(edgelen_npnew+0.001)
        wnp_npnewx1.unsqueeze_(1)

        psamplets = pb_npnewx2 * (1 - wnp_npnewx1) + pe_npnewx2 * wnp_npnewx1  # bilinear interpolation

        return psamplets

    def calc_min_distance(self, pya, pyb):
        # calc the min distance from pya to pyb
        pnuma, cnuma = pya.shape
        pnumb, cnumb = pyb.shape
        assert cnuma == 2 and cnumb == 2

        pya = pya.unsqueeze(1).expand((pnuma, pnumb, cnuma))
        pyb = pyb.unsqueeze(0).expand((pnuma, pnumb, cnuma))
        dis = torch.sum((pyb - pya) ** 2, dim=-1) # use l2 distance here, no sqrt
        dis_min = torch.min(dis, dim=1)[0] # the shape is [pnuma]
        return dis_min

    def calc_loss(self, py_pred, py_gt, score=None):
        pnum_pred, cnum_pred = py_pred.shape
        pnum_gt, cnum_gt = py_gt.shape
        assert cnum_pred == 2 and cnum_gt == 2

        dismin_p2g = self.calc_min_distance(py_pred,py_gt)
        dismin_g2p = self.calc_min_distance(py_gt,py_pred)
        if score is not None:
            score = torch.clip(score.squeeze(1),min=0.1,max=0.9)
            weight = (1-score)*2
            loss = torch.mean(dismin_p2g*weight) + torch.max(dismin_p2g*weight)
        else:
            loss = torch.mean(dismin_p2g) + torch.max(dismin_p2g)
        loss += torch.mean(dismin_g2p) + torch.max(dismin_g2p)
        return loss

    def forward(self, py_pred, py_gt, score=None):
        # py_gt = self.unifrom_resample(py_gt, self.pnum)
        # py_pred = self.unifrom_resample(py_pred, 200) # py_pred:[Ninst,Np,2]
        loss = self.calc_loss(py_pred,py_gt,score)
        return loss/200.0


class ScoreLoss(nn.Module):
    '''
    high score for better localization
    '''
    def __init__(self, cfg_score=None):
        super(ScoreLoss, self).__init__()
        self.pnum=128
        if cfg_score is not None:
            alpha = cfg_score['alpha']
            minshift = cfg_score['minshift'] if 'minshift' in cfg_score.keys() else 0
            use_xy_size = cfg_score['use_xy_size'] if 'use_xy_size' in cfg_score.keys() else False
        else:
            alpha = 0.05
            minshift = 0
            use_xy_size = False
        self.alpha = alpha
        self.minshift = minshift
        self.use_xy_size = use_xy_size
        assert self.minshift >= 0

    def unifrom_resample(self,poly, newpnum):
        pnum, cnum = poly.shape
        assert cnum == 2
        idxnext = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        polynext = poly[idxnext]  # shift edge to one index
        edgelen = torch.sqrt(torch.sum((polynext - poly) ** 2, dim=1))  # calc the edge len
        edgeidxsort = torch.argsort(edgelen)
        edgelen_cumsum = torch.cumsum(edgelen,dim=0)
        edgelentotal = edgelen.sum()
        edgelennew = edgelentotal/newpnum
        # find the begin and end point
        device = poly.device
        idx_npnew = torch.arange(newpnum).cuda(device)
        target_edgelen = edgelennew*idx_npnew.type(torch.float)
        target_edgelen_mat = target_edgelen.repeat((pnum,1)).transpose(1,0)
        diff = edgelen_cumsum.repeat((newpnum,1)) - target_edgelen_mat

        idx_np = torch.flip(torch.arange(1,pnum+1),dims=[0]).cuda(device)
        diff_mask = (diff>=0).type_as(idx_np) # find first one
        diff_idx = diff_mask * idx_np.repeat((newpnum,1))
        idxb_npnew = torch.argmax(diff_idx,dim=1) #[Np2,1]
        idxe_npnew = (idxb_npnew+1)%pnum
        # do interpolation
        init_zero = edgelen_cumsum[0:1]*0.0
        edgelen_cumsum = torch.cat((init_zero,edgelen_cumsum),0) # Np2+1
        pb_npnewx2 = torch.index_select(poly,0,idxb_npnew)
        pe_npnewx2 = torch.index_select(poly,0,idxe_npnew)
        taredgelen_npnew = target_edgelen
        edgelencumsum_npnew = torch.gather(edgelen_cumsum,0,idxb_npnew)
        edgelen_npnew = torch.gather(edgelen,0,idxb_npnew)
        wnp_npnewx1 = (taredgelen_npnew - edgelencumsum_npnew)/(edgelen_npnew+0.001)
        wnp_npnewx1.unsqueeze_(1)

        psamplets = pb_npnewx2 * (1 - wnp_npnewx1) + pe_npnewx2 * wnp_npnewx1  # bilinear interpolation

        return psamplets

    def calc_min_distance(self, pya, pyb, return_xy=False):
        # calc the min distance from pya to pyb
        pnuma, cnuma = pya.shape
        pnumb, cnumb = pyb.shape
        assert cnuma == 2 and cnumb == 2
        pya2 = pya.unsqueeze(1).expand((pnuma, pnumb, cnuma))
        pyb2 = pyb.unsqueeze(0).expand((pnuma, pnumb, cnuma))
        dis = torch.sqrt(torch.sum((pyb2 - pya2) ** 2, dim=-1))
        dis_min, idx_min = torch.min(dis, dim=1) # the shape is [pnuma]
        if return_xy:
            pyb_a = pyb[idx_min,:]
            diff = pya - pyb_a
            x_diff = diff[:,0]
            y_diff = diff[:,1]
            return dis_min, x_diff, y_diff
        else:
            return dis_min

    def calc_dis_size(self,py):
        pnum, cnum = py.shape
        x_max, x_min = py[:,0].max(),py[:,0].min()
        y_max, y_min = py[:,1].max(),py[:,1].min()
        dis_size = torch.sqrt((x_max-x_min)**2+(y_max-y_min)**2)
        return dis_size

    def calc_xy_size(self,py):
        pnum, cnum = py.shape
        x_max, x_min = py[:,0].max(),py[:,0].min()
        y_max, y_min = py[:,1].max(),py[:,1].min()
        # dis_size = torch.sqrt((x_max-x_min)**2+(y_max-y_min)**2)
        x_size = x_max - x_min
        y_size = y_max - y_min
        return x_size,y_size

    def calc_score_gt(self, py_pred, py_gt):
        pnum_pred, cnum_pred = py_pred.shape
        pnum_gt, cnum_gt = py_gt.shape
        assert cnum_pred == 2 and cnum_gt == 2
        if not self.use_xy_size:
            dismin_p2g = self.calc_min_distance(py_pred, py_gt)
            dis_size = self.calc_dis_size(py_gt)
            shift = max(self.alpha*dis_size,self.minshift)
            s_gt = 1-torch.sigmoid(dismin_p2g-shift)
        else:
            x_size, y_size = self.calc_xy_size(py_gt)
            x_shift = max(self.alpha*x_size,self.minshift)
            y_shift = max(self.alpha*y_size,self.minshift)
            dismin_p2g, x_diff, y_diff = self.calc_min_distance(py_pred, py_gt,return_xy=True)
            s_gtx = 1-torch.sigmoid(x_diff.abs()-x_shift)
            s_gty = 1-torch.sigmoid(y_diff.abs()-y_shift)
            s_gt = torch.cat((s_gtx.unsqueeze(1),s_gty.unsqueeze(1)),dim=1)
            s_gt = torch.min(s_gt,dim=1)[0]
        return s_gt

    def forward(self, s_pred, py_pred, py_gt):
        # py_gt = self.unifrom_resample(py_gt, self.pnum)
        # py_pred = self.unifrom_resample(py_pred, 200) # py_pred:[Ninst,Np,2]
        s_gt = self.calc_score_gt(py_pred,py_gt)
        if s_pred == None:
            return s_gt
        loss = F.smooth_l1_loss(s_pred.squeeze(1),s_gt)
        # return loss
        return loss, s_gt



def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    """
    feat: [B,C,H,W]
    ind: [B,Ninstb]
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3)) # [B,HW,C]
    feat = _gather_feat(feat, ind)
    return feat



class IndL1Loss1d(nn.Module):
    def __init__(self, type='l1') -> object:
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    # pths = [pth.split('.')[0] for pth in os.listdir(model_dir)]
    pths = []
    for pth in os.listdir(model_dir):
        if pth.split('.')[0] != 'EarlyStop':
            pths.append(int(pth.split('.')[0]))
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))
    print(f'Save model of epoch{epoch}.')
    # # remove previous pretrained model if the number of models is too big
    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    # if len(pths) <= 200:
    #     return
    # os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    # pths = [pth[:-4] for pth in os.listdir(model_dir) if 'pth' in pth]
    pths = []
    pths_temp = glob.glob(osp.join(model_dir,'*.pth'))
    for pth in iter(pths_temp):
        pth = osp.split(pth)[-1]
        if pth.split('.')[0] != 'EarlyStop':
            pths.append(int(pth.split('.')[0]))
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1

def load_network_given_ckpt(net, ckpt_path, strict=True):

    if not os.path.exists(ckpt_path):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(ckpt_path))
    pretrained_model = torch.load(ckpt_path)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1

def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, optim, scheduler, recorder, model_dir,
                 patience=7, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path = path # dont use
        self.trace_func = trace_func
        self.optim = optim
        self.scheduler = scheduler
        self.recorder = recorder
        self.model_dir = model_dir

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_model(self, net, epoch):
        '''follow above save_model function'''
        os.system('mkdir -p {}'.format(self.model_dir))
        torch.save({
            'net': net.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'recorder': self.recorder.state_dict(),
            'epoch': epoch
        }, os.path.join(self.model_dir, 'EarlyStop.pth'))

    def save_checkpoint(self, val_loss, model,epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.save_model(model, epoch)
        self.val_loss_min = val_loss