import random

import torch
import numpy as np
from lib.utils.ssnake import snake_decode, snake_config
from lib.csrc.extreme_utils import _ext as extreme_utils
from lib.utils import data_utils
import torch.nn.functional as F


def collect_training(poly, ct_01):
    # concat polys from B imgs (namely, one batch)
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly


def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].type(torch.bool)
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config.box_iou) * (score > snake_config.confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind]

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1]
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind


def prepare_training_box(output, batch, init):
    box = output['detection'][..., :4]
    score = output['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].type(torch.bool)
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([snake_decode.get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = snake_decode.get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})

def prepare_training(ret, batch):
    ct_01 = batch['ct_01'].type(torch.bool)
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)}) # concat to Ninst of all batches
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    # 4py_ind & py_ind: [Ninst of all this batch] 0000...111...222..BatchBatchBatch
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    if snake_config.train_pred_box:
        prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def uniform_resample(poly, newpnum):
    pnum, cnum = poly.shape
    assert cnum == 2
    idxnext = (torch.arange(pnum, dtype=torch.long) + 1) % pnum
    polynext = poly[idxnext]  # shift edge to one index
    edgelen = torch.sqrt(torch.sum((polynext - poly) ** 2, dim=1))  # calc the edge len
    # edgeidxsort = torch.argsort(edgelen)
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

def uniform_resample_polys(polys, Np2):
    Ninst, Np, cnum = polys.shape
    assert cnum == 2
    device = polys.device
    if Ninst == 0:
        return torch.zeros((Ninst,Np2,cnum),dtype=polys.dtype,device=device)

    polysnext = torch.roll(polys,1,dims=1)  # shift edge to one index
    edgelen = torch.sqrt(torch.sum((polysnext - polys) ** 2, dim=2))  # calc the edge len [Ninst,Np]
    edgelen_cumsum = torch.cumsum(edgelen,dim=1) #[Ninst,Np]
    edgelentotal = edgelen.sum(dim=1,keepdim=True) #[Ninst,1]
    edgelennew = edgelentotal/Np2
    # find the begin and end point
    idx_npnew = torch.arange(Np2).unsqueeze(0).to(device) # [1,Np']
    target_edgelen = edgelennew*idx_npnew.type(torch.float) #[Ninst,Np']
    target_edgelen_mat = target_edgelen.unsqueeze(-1)  #[Ninst,Np',Np]??
    src_edgelen = edgelen_cumsum[:,list(range(-1,Np-1))]; src_edgelen[:,0] = 0;

    diff = target_edgelen_mat - src_edgelen.unsqueeze(1) #[Ninst,Np',Np]
    idx_np = torch.arange(1,Np+1).to(device) #[Np]
    diff_mask = (diff>=0).type_as(idx_np) # find first one, [Ninst,Np',Np]
    diff_idx = diff_mask * idx_np.unsqueeze(0).unsqueeze(0)
    idxb_npnew = torch.argmax(diff_idx,dim=2) #[Ninst,Np,1]
    idxe_npnew = (idxb_npnew+1)%Np

    # do interpolation
    diff_b = diff.gather(2,idxb_npnew.unsqueeze(-1))
    edgelen_b =  edgelen.unsqueeze(1).repeat((1,Np2,1)).gather(2,idxb_npnew.unsqueeze(-1)) # [Ninst,Np']
    weight = diff_b / edgelen_b # [Ninst,Np']
    polys_for_idx = polys.unsqueeze(1).repeat((1,Np2,1,1)) #[Ninst,Np',Np,2]
    idxb_npnewd2 = idxb_npnew.unsqueeze(-1).unsqueeze(-1).repeat((1,1,1,2))
    pb_npnewd2 = polys_for_idx.gather(2,idxb_npnewd2).squeeze(2)# [Ninst,Np',2]
    idxe_npnewd2 = idxe_npnew.unsqueeze(-1).unsqueeze(-1).repeat((1,1,1,2))
    pe_npnewd2 = polys_for_idx.gather(2,idxe_npnewd2).squeeze(2)
    # if weight.min()<0 or weight.max() > 1:
    #     print('1')
    psamplets = pb_npnewd2 * (1 - weight) + pe_npnewd2 * weight  # bilinear interpolation
    return psamplets

def uniform_resample_polys_np(polys, newpnum):
    Ninst, pnum, cnum = polys.shape
    assert cnum == 2
    if Ninst == 0:
        return np.zeros((Ninst,newpnum,cnum),dtype=polys.dtype)

    idxnext = (np.arange(pnum, dtype=np.long) + 1) % pnum
    polysnext = polys[:,idxnext,:]  # shift edge to one index
    edgelen = np.sqrt(np.sum((polysnext - polys) ** 2, axis=2))  # calc the edge len
    edgelen_cumsum = np.cumsum(edgelen,axis=1) #[Ninst,Np]
    edgelentotal = edgelen.sum(axis=1,keepdims=True) #[Ninst,1]
    edgelennew = edgelentotal/newpnum
    # find the begin and end point
    idx_npnew = np.arange(newpnum)[np.newaxis,...] # [1,Np']
    target_edgelen = edgelennew*idx_npnew.astype(np.float) #[Ninst,Np']
    target_edgelen_mat = target_edgelen[...,np.newaxis]  #[Ninst,Np',Np]??
    src_edgelen = edgelen_cumsum[:,list(range(-1,pnum-1))]; src_edgelen[:,0] = 0;

    diff = target_edgelen_mat - src_edgelen[:,np.newaxis,...] #[Ninst,Np',Np]
    idx_np = np.arange(1,pnum+1) #[Np]
    diff_mask = (diff>=0).astype(idx_np.dtype) # find first one, [Ninst,Np',Np]
    diff_idx = diff_mask * idx_np[np.newaxis,np.newaxis,...]
    idxb_npnew = np.argmax(diff_idx,axis=2) #[Ninst,Np,1]
    idxe_npnew = (idxb_npnew+1)%pnum

    # do interpolation
    diff_b = np.take_along_axis(diff, idxb_npnew[..., np.newaxis], 2)
    edgelen_b =  np.take_along_axis(np.repeat(edgelen[:,np.newaxis,...],newpnum,axis=1),
                                    idxb_npnew[...,np.newaxis],
                                    2) # [Ninst,Np']
    weight = diff_b / edgelen_b # [Ninst,Np']
    polys_for_idx = np.repeat(polys[:,np.newaxis,...],
                              newpnum,axis=1) #[Ninst,Np',Np,2]
    idxb_npnewd2 = np.repeat(idxb_npnew[...,np.newaxis,np.newaxis],2,axis=3)
    pb_npnewd2 = np.take_along_axis(polys_for_idx,idxb_npnewd2,2)[:,:,0,...]# [Ninst,Np',2]
    idxe_npnewd2 = np.repeat(idxe_npnew[...,np.newaxis,np.newaxis],2,axis=3)
    pe_npnewd2 = np.take_along_axis(polys_for_idx,idxe_npnewd2,2)[:,:,0,...]# [Ninst,Np',2]

    psamplets = pb_npnewd2 * (1 - weight) + pe_npnewd2 * weight  # bilinear interpolation

    return psamplets

def gather_numpy(array, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = array.shape[:dim] + array.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(array, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


def box2poly_acw_tensor(bbox, Np):
    """
    generate polys clockwise from bounding box.
    bbox: [N,4]: l,r,t,d
    return:
        polys: [N,Np,2]
    """
    assert Np % 4 == 0
    Ne = (Np // 4) - 1
    device = bbox.device
    l_coord = bbox[:, 0].unsqueeze(-1).unsqueeze(-1)
    r_coord = bbox[:, 1].unsqueeze(-1).unsqueeze(-1)
    t_coord = bbox[:, 2].unsqueeze(-1).unsqueeze(-1)
    d_coord = bbox[:, 3].unsqueeze(-1).unsqueeze(-1)
    ids = torch.arange(1, Ne + 1, step=1, device=device, dtype=torch.float32)  # 1...Ne
    ids = ids.unsqueeze(0).unsqueeze(-1)  # [1,Ne,1]
    ld_coord = torch.concat((l_coord, d_coord), dim=2)  # [N,1,2]
    rd_coord = torch.concat((r_coord, d_coord), dim=2)
    rt_coord = torch.concat((r_coord, t_coord), dim=2)
    lt_coord = torch.concat((l_coord, t_coord), dim=2)
    edge_coord = l_coord + (r_coord - l_coord) / (Ne + 1) * ids  # [N,31,1]
    line_t = torch.concat((edge_coord, t_coord.expand_as(edge_coord)), dim=2)
    edge_coord = t_coord + (d_coord - t_coord) / (Ne + 1) * ids
    line_r = torch.torch.concat((r_coord.expand_as(edge_coord), edge_coord), dim=2)
    edge_coord = r_coord + (l_coord - r_coord) / (Ne + 1) * ids
    line_d = torch.torch.concat((edge_coord, d_coord.expand_as(edge_coord)), dim=2)
    edge_coord = d_coord + (t_coord - d_coord) / (Ne + 1) * ids
    line_l = torch.torch.concat((l_coord.expand_as(edge_coord), edge_coord), dim=2)
    polys = torch.concat((lt_coord, line_t, rt_coord, line_r, rd_coord, line_d, ld_coord, line_l), dim=1)
    polys = smooth_corners(polys, method='interp')

    return polys
def uniformsample_slow(bbox, Np, w, h):
    """
    generate polys clockwise from bounding box.
    bbox: [N,4]: l,r,t,d
    return:
        polys: [N,Np,2]
    """
    assert Np % 4 == 0
    device = bbox.device
    Ninst = bbox.shape[0]
    if Ninst == 0:
        polys = torch.zeros((0,Np,2),device=device)
        return polys
    bbox_coord = torch.zeros((Ninst, 4, 2), device=device)  # shape:[Ninst,4,2]
    bbox_coord[:, 0, 0] = bbox[:, 0]  # l
    bbox_coord[:, 0, 1] = bbox[:, 2]  # t
    bbox_coord[:, 1, 0] = bbox[:, 1]  # r
    bbox_coord[:, 1, 1] = bbox[:, 2]  # t
    bbox_coord[:, 2, 0] = bbox[:, 1]  # r
    bbox_coord[:, 2, 1] = bbox[:, 3]  # d
    bbox_coord[:, 3, 0] = bbox[:, 0]  # l
    bbox_coord[:, 3, 1] = bbox[:, 3]  # d
    bbox_coord[..., 0] = torch.clamp(bbox_coord[..., 0], 0, w)
    bbox_coord[..., 1] = torch.clamp(bbox_coord[..., 1], 0, h)
    pnum = Np
    polys = []
    for i in range(Ninst):
        box_py = bbox_coord[i, ...]
        poly = uniform_resample(box_py, newpnum=pnum)
        polys.append(poly)
    polys = torch.stack(polys, dim=0)
    return polys

def sequence_random_rolling(pys):
    """
    random roll sequence
    pys: [N,Np,2]
    return:
        pys: [N,Np,2]
    """
    Ninst = len(pys)
    if Ninst == 0:
        return pys
    Np = pys.shape[1]
    # pys = torch.roll(pys, shifts=int(shift), dims=1)
    shifts = []
    for i in range(Ninst):
        shift = np.random.randint(low=0,high=Np,size=1,dtype=int)[0]
        pys[i,...] = torch.roll(pys[i,...],shifts=int(shift),dims=0)
        shifts.append(shift)
    return pys

def uniformsample_quick(bbox, Np, w, h):
    """
    generate polys clockwise from bounding box.
    bbox: [N,4]: l,r,t,d
    return:
        polys: [N,Np,2]
    """
    assert Np % 4 == 0
    device = bbox.device
    Ninst = bbox.shape[0]
    if Ninst == 0:
        polys = torch.zeros((0,Np,2),device=device)
        return polys
    bbox_coord = torch.zeros((Ninst, 4, 2), device=device)  # shape:[Ninst,4,2]
    bbox_coord[:, 0, 0] = bbox[:, 0]  # l
    bbox_coord[:, 0, 1] = bbox[:, 2]  # t
    bbox_coord[:, 1, 0] = bbox[:, 1]  # r
    bbox_coord[:, 1, 1] = bbox[:, 2]  # t
    bbox_coord[:, 2, 0] = bbox[:, 1]  # r
    bbox_coord[:, 2, 1] = bbox[:, 3]  # d
    bbox_coord[:, 3, 0] = bbox[:, 0]  # l
    bbox_coord[:, 3, 1] = bbox[:, 3]  # d
    bbox_coord[..., 0] = torch.clamp(bbox_coord[..., 0], 0, w)
    bbox_coord[..., 1] = torch.clamp(bbox_coord[..., 1], 0, h)
    pnum = Np
    polys = uniform_resample_polys(bbox_coord,newpnum=pnum)
    return polys

def uniformsample_polys_quick(pys, Np, w=None, h=None):
    """
    generate polys clockwise from bounding box.
    pys: [N,Np,2]
    return:
        polys: [N,Np,2]
    """
    assert Np % 4 == 0
    device = pys.device
    Ninst = pys.shape[0]
    if Ninst == 0:
        polys = torch.zeros((0,Np,2),device=device)
        return polys
    if w is not None:
        pys[..., 0] = torch.clamp(pys[..., 0], 0, w)
    if h is not None:
        pys[..., 1] = torch.clamp(pys[..., 1], 0, h)
    pnum = Np
    polys = uniform_resample_polys(pys,newpnum=pnum)
    return polys


def mid_upsample(polys):
    Ninst, pnum, cnum = polys.shape
    assert cnum == 2
    dtype = polys.dtype;device = polys.device
    pnum_up = pnum * 2
    polys_up = torch.zeros((Ninst, pnum_up, cnum), device=device, dtype=dtype)
    if Ninst == 0:
        return polys_up

    polys_next = torch.cat([polys[:,1:,:],polys[:,0:1,:]],dim=1)
    polys_mid = (polys+polys_next)/2
    polys_up[:,0:pnum_up:2,:] = polys
    polys_up[:,1:pnum_up:2,:] = polys_mid

    return polys_up

def prepare_training_myshape_gt(batch,Np):
    ct = batch['ct_coord']
    ct_01 = batch['ct_01'].type(torch.bool)
    Ninst = ct_01.sum()
    ct_num = batch['meta']['ct_num']
    # inst_ind: [Ninst of all this batch] 0000...111...222..BatchBatchBatch
    inst_ind = torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)
    batch.update({'py_ind':inst_ind}) # shape[Ninst,Np,2]
    # for ct points
    collect = collect_training
    ct_gt = collect(ct, ct_01).unsqueeze(1)  # shape[Ninst,2]
    wh_gt = batch['wh']
    wh_gt = collect(wh_gt, ct_01)
    device = wh_gt.device
    scale_range=(0.8,1.1)
    scale = torch.rand((1),device=device)*(scale_range[1]-scale_range[0])+(scale_range[0])
    wh_fin = wh_gt
    # wh_fin = wh_fin*scale
    ct_gt = ct_gt.squeeze(1)
    # generate init_poly
    bbox = torch.zeros((Ninst,4),device=device) # shape:[Ninst,4]
    bbox[:, 0] = ct_gt[:,0]-wh_fin[:,0]/2 # l
    bbox[:, 1] = ct_gt[:,0]+wh_fin[:,0]/2 # r
    bbox[:, 2] = ct_gt[:,1]+wh_fin[:,1]/2 # t
    bbox[:, 3] = ct_gt[:,1]-wh_fin[:,1]/2 # d
    bbox_coord = torch.zeros((Ninst,4,2),device=device) #shape:[Ninst,4,2]
    bbox_coord[:,0,0] = bbox[:,0]#l
    bbox_coord[:,0,1] = bbox[:,2]#t
    bbox_coord[:,1,0] = bbox[:,1]#r
    bbox_coord[:,1,1] = bbox[:,2]#t
    bbox_coord[:,2,0] = bbox[:,1]#r
    bbox_coord[:,2,1] = bbox[:,3]#d
    bbox_coord[:,3,0] = bbox[:,0]#l
    bbox_coord[:,3,1] = bbox[:,3]#d
    pnum = Np
    init_pys = []
    for i in range(Ninst):
        box_py = bbox_coord[i,...]
        init_py = uniform_resample(box_py, newpnum=Np)
        init_pys.append(init_py)
    if len(init_pys) == 0:
        init_pys = torch.zeros((0,Np,2),device=device)
    else:
        init_pys = torch.stack(init_pys,dim=0)

    can_init_pys = img_poly_to_can_poly(init_pys)
    init = {}
    init.update({'init_py':init_pys,'can_init_py':can_init_pys}) # shape[Ninst,Np,2]
    return init

def calc_min_distance(pya, pyb):
    """
    pya, pyb: [Np, 2]
    return:
    dis_min: [Np]
    """
    # calc the min distance from pya to pyb
    pnuma, cnuma = pya.shape
    pnumb, cnumb = pyb.shape
    assert cnuma == 2 and cnumb == 2

    pya = pya.unsqueeze(1).expand((pnuma, pnumb, cnuma))
    pyb = pyb.unsqueeze(0).expand((pnuma, pnumb, cnuma))
    dis = torch.sum((pyb - pya) ** 2, dim=-1) # use l2 distance here, no sqrt
    dis_min = torch.min(dis, dim=1)[0] # the shape is [pnuma]
    return dis_min

def calc_min_distance_parallel(pya, pyb, use_b_uniform = True):
    """
    pya, pyb: [Ninst, Npa, 2], [Ninst, Npb, 2], normally, a:pred, b:gt
    return:
    dis_min: [Ninst, Npa]
    """
    # calc the min distance from pya to pyb
    Ninst, pnuma, cnuma = pya.shape
    Ninst, pnumb, cnumb = pyb.shape
    assert cnuma == 2 and cnumb == 2
    assert pnumb == pnuma
    if use_b_uniform:
        pyb = uniform_resample_polys(pyb,pnumb)
    pya_ori, pyb_ori = pya.clone(), pyb.clone()
    pya = pya_ori.unsqueeze(-2).expand((Ninst, pnuma, pnumb, cnuma))
    pyb = pyb_ori.unsqueeze(-3).expand((Ninst, pnuma, pnumb, cnumb))
    dis = torch.sqrt(torch.sum((pyb - pya) ** 2, dim=-1)) # use l2 distance here, no sqrt
    dis_min, idx_min= torch.min(dis, dim=-1) # the shape is [pnuma]
    idx_min = idx_min.unsqueeze(-1).expand(Ninst,pnumb,cnumb)
    pyb_a = torch.gather(pyb_ori,1,idx_min) # b but shape as a
    v_b2a = pya_ori - pyb_a
    return dis_min, v_b2a, pyb_a

def rescale_polys(polys,scale_facor=1.0):
    """
    rescale polys by scale
    polys:[Ninst, Np, 2]
    """
    Ninst = len(polys)
    if Ninst < 1:
        return polys
    polys_x = polys[...,0]
    x_min = torch.min(polys_x,dim=1,keepdim=True)[0]
    x_max = torch.max(polys_x,dim=1,keepdim=True)[0]
    x_ct = (x_min + x_max)/2
    polys_x_scl = (polys_x - x_ct)*scale_facor + x_ct
    polys_y = polys[...,1]
    y_min = torch.min(polys_y, dim=1,keepdim=True)[0]
    y_max = torch.max(polys_y, dim=1,keepdim=True)[0]
    y_ct = (y_min + y_max) / 2
    polys_y_scl = (polys_y - y_ct) * scale_facor + y_ct
    polys_scl = torch.cat([polys_x_scl.unsqueeze(-1), polys_y_scl.unsqueeze(-1)],dim=2)
    return polys_scl

def prepare_training_myshape(output, batch,Np,w,h, cfg_box_aug=None):
    ct = batch['ct_coord']
    ct_01 = batch['ct_01'].type(torch.bool)
    Ninst = ct_01.sum()
    ct_num = batch['meta']['ct_num']
    # inst_ind: [Ninst of all this batch] 0000...111...222..BatchBatchBatch
    inst_ind = torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)
    batch.update({'py_ind':inst_ind}) # shape[Ninst,Np,2]
    # for ct points
    collect = collect_training
    ct_gt = collect(ct, ct_01).unsqueeze(1)  # shape[Ninst,2]
    wh_map = output['wh']
    w_whmap, h_whmap = wh_map.shape[-2:]
    wh_out = get_gcn_feature(wh_map, ct_gt, inst_ind, w_whmap,h_whmap)  # [Ninst,2,Np=1]
    wh_out = wh_out.squeeze(-1) # [Ninst,2]
    wh_gt = batch['wh']
    wh_gt = collect(wh_gt, ct_01)
    device = wh_gt.device
    if cfg_box_aug is not None:
        use_box_aug = cfg_box_aug['use']
        scale_range = list(cfg_box_aug['scale_range'])
        trust_out = cfg_box_aug['trust_out']
        shift_range = cfg_box_aug['shift_range'] if 'shift_range' in cfg_box_aug.keys() else (0,0)
        if use_box_aug:
            scale = torch.rand((Ninst,2),device=device)*(scale_range[1]-scale_range[0])+(scale_range[0])
            wh_fin = wh_gt*(1-trust_out)+wh_out*trust_out
            wh_fin = wh_fin*scale
            ct_gt = ct_gt.squeeze(1)
            shift_scale = torch.rand((Ninst,2),device=device)*(shift_range[1]-shift_range[0])+(shift_range[0]) # the ct shift from the ct_gt, w.r.t wh_gt #map relation:(-w/2,w/2)=(-1,1)
            ct_shift = shift_scale * wh_gt/2
            ct = ct_gt + ct_shift # use ct aug
            # ct = ct_gt
        else:
            wh_fin = wh_gt
            ct = ct_gt
    # generate init_poly
    bbox = torch.zeros((Ninst,4),device=device) # shape:[Ninst,4]
    bbox[:, 0] = ct[:,0]-wh_fin[:,0]/2 # l
    bbox[:, 1] = ct[:,0]+wh_fin[:,0]/2 # r
    bbox[:, 2] = ct[:,1]+wh_fin[:,1]/2 # t
    bbox[:, 3] = ct[:,1]-wh_fin[:,1]/2 # d
    if cfg_box_aug is not None:
        cfg_sample_method = cfg_box_aug['sample_method']
    else:
        cfg_sample_method = 'uniform'
    assert cfg_sample_method in ('box', 'uniform', 'uniform_quick')
    if cfg_sample_method == 'uniform':
        init_pys = uniformsample_slow(bbox,Np,w,h)
    elif cfg_sample_method == 'uniform_quick':
        init_pys = uniformsample_quick(bbox,Np,w,h)
    else:
        init_pys = box2poly_acw_tensor(bbox,Np)

    if len(init_pys) == 0:
        init_pys = torch.zeros((0,Np,2),device=device)

    can_init_pys = img_poly_to_can_poly(init_pys)
    init = {}
    init.update({'init_py':init_pys,'can_init_py':can_init_pys}) # shape[Ninst,Np,2]
    return init

def prepare_testing_myshape(output, Np, w, h, cfg_box_aug=None, scale_factor=1.0):
    ct = output['ct']
    B,Ninst_b = ct.shape[:2]
    Ninst = B*Ninst_b
    ct = ct.reshape(Ninst,2)
    # generate init_poly
    detection = output['detection'].reshape(Ninst,6)
    device = ct.device
    bbox = torch.zeros((Ninst,4),device=device) # shape:[Ninst,4]
    bbox[:, 0] = detection[...,0] #l
    bbox[:, 1] = detection[...,2] # r
    bbox[:, 2] = detection[...,3] # t
    bbox[:, 3] = detection[...,1] # d
    if scale_factor != 1.0:
        ct_x = (bbox[..., 0] + bbox[..., 1]) / 2
        ct_y = (bbox[..., 2] + bbox[..., 3]) / 2
        bbox[..., 0] = ct_x + (bbox[..., 0] - ct_x)*scale_factor
        bbox[..., 1] = ct_x + (bbox[..., 1] - ct_x)*scale_factor
        bbox[..., 2] = ct_y + (bbox[..., 2] - ct_y)*scale_factor
        bbox[..., 3] = ct_y + (bbox[..., 3] - ct_y)*scale_factor
    assert h == w
    bbox = torch.clamp(bbox, min=0, max=h)
    if cfg_box_aug is not None:
        cfg_sample_method = cfg_box_aug['sample_method']
    else:
        cfg_sample_method = 'uniform'
    assert cfg_sample_method in ('box', 'uniform', 'uniform_quick','fft_uniform')
    if cfg_sample_method == 'uniform':
        init_pys = uniformsample_slow(bbox,Np,w,h)
    elif cfg_sample_method == 'uniform_quick':
        init_pys = uniformsample_quick(bbox,Np,w,h)
    else:
        init_pys = box2poly_acw_tensor(bbox,Np)
    can_init_pys = img_poly_to_can_poly(init_pys)
    init={}
    init.update({'init_py':init_pys,'can_init_py':can_init_pys}) # shape[Ninst,Np,2]
    init_ind = [torch.Tensor(Ninst_b*[i]) for i in range(B)]
    init_ind = torch.cat(init_ind)
    init.update({'ind':init_ind})
    return init

def prepare_testing_myshape_gtct(batch,output, Np):
    '''
    use gt center
    '''
    ct = batch['ct_coord']
    ct_01 = batch['ct_01'].type(torch.bool)
    inst_ind = batch['py_ind']
    B, Ninst_b = ct.shape[:2]
    Ninst = B * Ninst_b
    device = ct.device
    # for ct points
    collect = collect_training
    ct_gt = collect(ct, ct_01).unsqueeze(1)  # shape[Ninst,2]

    wh_map = output['wh']
    w_whmap, h_whmap = wh_map.shape[-2:]
    wh_out = get_gcn_feature(wh_map, ct_gt, inst_ind, w_whmap, h_whmap)  # [Ninst,2,Np=1]
    wh_out = wh_out.squeeze(-1)
    wh_fin = wh_out # choose to use wh gt or not
    ct_gt = ct_gt.squeeze(1)
    # generate init_poly
    bbox = torch.zeros((Ninst, 4), device=device)  # shape:[Ninst,4]
    bbox[:, 0] = ct_gt[:, 0] - wh_fin[:, 0] / 2  # l
    bbox[:, 1] = ct_gt[:, 0] + wh_fin[:, 0] / 2  # r
    bbox[:, 2] = ct_gt[:, 1] + wh_fin[:, 1] / 2  # t
    bbox[:, 3] = ct_gt[:, 1] - wh_fin[:, 1] / 2  # d
    bbox_coord = torch.zeros((Ninst, 4, 2), device=device)  # shape:[Ninst,4,2]
    bbox_coord[:, 0, 0] = bbox[:, 0]  # l
    bbox_coord[:, 0, 1] = bbox[:, 2]  # t
    bbox_coord[:, 1, 0] = bbox[:, 1]  # r
    bbox_coord[:, 1, 1] = bbox[:, 2]  # t
    bbox_coord[:, 2, 0] = bbox[:, 1]  # r
    bbox_coord[:, 2, 1] = bbox[:, 3]  # d
    bbox_coord[:, 3, 0] = bbox[:, 0]  # l
    bbox_coord[:, 3, 1] = bbox[:, 3]  # d
    init_pys = []
    for i in range(Ninst):
        box_py = bbox_coord[i, ...]
        init_py = uniform_resample(box_py, newpnum=Np)
        init_pys.append(init_py)
    if len(init_pys) == 0:
        init_pys = torch.zeros((0, Np, 2), device=device)
    else:
        init_pys = torch.stack(init_pys, dim=0)
    can_init_pys = img_poly_to_can_poly(init_pys)
    init = {}
    init.update({'init_py': init_pys, 'can_init_py': can_init_pys})  # shape[Ninst,Np,2]
    init_ind = inst_ind
    init.update({'ind': init_ind})
    return init

def PolyAug(polys,h,w,scale_range=(0.7,1.1)):
    """
    polys:[Ninst,Np,2]
    scale_range:list or tuple, len 2
    """
    if len(polys) == 0:
        return torch.zeros_like(polys)
    Ninst,Np,_ = polys.size()
    polys = polys.detach()
    pl = polys[...,0].min(dim=1)[0]
    pr = polys[...,0].max(dim=1)[0]
    pd = polys[..., 1].min(dim=1)[0]
    pt = polys[..., 1].max(dim=1)[0]
    ct_w = ((pl+pr)/2).unsqueeze(-1).unsqueeze(-1)
    ct_h = ((pt+pd)/2).unsqueeze(-1).unsqueeze(-1)
    #[Ninst,1,2]
    cts = torch.cat([ct_w,ct_h],dim=-1)
    device = polys.device
    scale = torch.rand((Ninst, 1, 2), device=device) * (scale_range[1] - scale_range[0]) + (scale_range[0])
    polys_aug = cts+(polys-cts)*scale
    polys_aug[...,0] = torch.clamp(polys_aug[...,0],0,w)
    polys_aug[...,1] = torch.clamp(polys_aug[...,1],0,h)

    return polys_aug


def obtain_inst_feature(ct_map,feat_map,img_poly,ind):
    '''
    use cropped ct_hm * feature as inst feature
    '''
    ct_map = torch.sigmoid(ct_map)
    feat_map = ct_map.expand_as(feat_map) * feat_map
    assert img_poly.shape[-1] == 2 and len(img_poly.shape) == 3
    B, C, h, w = feat_map.shape
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    l = img_poly[:, :, 0].min(dim=1)[0].detach().cpu().numpy() # [Ninst,Np,2]
    r = img_poly[:, :, 0].max(dim=1)[0].detach().cpu().numpy()
    b = img_poly[:, :, 1].min(dim=1)[0].detach().cpu().numpy()
    t = img_poly[:, :, 1].max(dim=1)[0].detach().cpu().numpy()
    xs = np.linspace(l,r,num=16,endpoint=True)
    xs =torch.tensor(xs).type_as(img_poly)
    ys = np.linspace(b,t,num=16,endpoint=True)
    ys =torch.tensor(ys).type_as(img_poly) #[16,Ninst]
    device = feat_map.device
    num_py = len(img_poly)
    mesh_coord = torch.zeros((num_py,16*16,2),device=device)
    for i in range(num_py):
        _xs = xs[:,i]
        _ys = ys[:,i]
        _mesh_coord = torch.cartesian_prod(_xs,_ys)
        mesh_coord[i,:,:] = _mesh_coord
    ind = ind.type(torch.int)

    inst_feature = torch.zeros((num_py,C),device=device)
    for i in range(B):
        # _l, _r, _b, _t = l[i],r[i],b[i],t[i]
        _feat_map = feat_map[i:i+1] #[1,C,H,W]
        _coords = mesh_coord[ind==i].unsqueeze(0) #[1,Ninst',Nlin*N,2]
        _feature = torch.nn.functional.grid_sample(_feat_map, _coords)[0].permute(1, 0, 2) #[Ninst,C,Nlin*Nlin]
        inst_feature[ind==i] = _feature.mean(dim=2)
    return inst_feature

def fuse_inst_feat(fuse_layer,inst_feat,init_feature, score = None, res=True):
    Ninst,_,Np = init_feature.shape
    Cinst = inst_feat.shape[1]
    # init_feature = fuse_layer(init_feature)
    # init_feature = init_feature + inst_feat.unsqueeze(-1).expand((Ninst,Cinst,Np))
    init_feature_ori = init_feature
    init_feature = torch.cat([init_feature, inst_feat.unsqueeze(-1).expand((Ninst,Cinst,Np))], dim=1)
    if score is not None:
        score = score.permute((0,2,1)) # [Ninst,Np,1] -> [Ninst,1,Np]
        init_feature = torch.cat([init_feature, score], dim=1)
    init_feature = fuse_layer(init_feature)
    if res:
        init_feature = init_feature + init_feature_ori
    return init_feature



def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    # img_poly: shape[Nisnt,Np,2]
    # cnn_feature: shape[B,C,H,W]
    # ind: 0000...111...22...333...(B-1)(B-1),len(Ninst)
    assert img_poly.shape[-1]==2 and len(img_poly.shape)==3
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device) # [Ninst,Nfeature,Np]
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0) # [1,Ninst,Np,2]
        poly = poly.type_as(cnn_feature)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature

    return gcn_feature


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)



def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly

def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly


def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly

def smooth_corners(polys, method='interp'):
    """
    to smoothen refined results from function box2poly_acw_tensor
    polys:[Ninst,Np,2]
    """
    Ninst, pnum, cnum = polys.shape
    if pnum == 0:
        pnum=snake_config.poly_num
    assert cnum == 2
    device = polys.device
    if Ninst == 0:
        return torch.zeros((Ninst,pnum,cnum),dtype=polys.dtype,device=device)
    assert pnum % 4 == 0
    Ne = (pnum // 4) - 1 #eg:31
    idx_list = [0,Ne+1,2*(Ne+1),3*(Ne+1)]
    if method == 'interp':
        for idx in iter(idx_list):
            polys[:,idx,:] = (polys[:,idx-1,:] + polys[:,idx+1,:])/2

    return polys



