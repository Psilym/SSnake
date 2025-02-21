from torch.utils.data.dataloader import default_collate
import torch
import numpy as np

def ssnake_collator(batch):
    from lib.utils.ssnake import snake_config

    ret = {'inp': default_collate([b['inp'] for b in batch])}

    if 'mask' in batch[0].keys():
        ret.update({'mask': default_collate([b['mask'] for b in batch])})

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})
    if 'speed_eval' in meta:
        return ret

    # instance polygon # for test
    if 'test' in meta:
        py = default_collate([b['py'] for b in batch])
        ret.update({'py': py})

    if 'test' in meta:
        return ret

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])
    edge_hm = default_collate([b['edge_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    wh = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_coord = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.bool)

    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    if ct_num != 0:
        wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
        ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
        ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
        ct_coord[ct_01] = torch.Tensor(sum([b['ct_coord'] for b in batch], []))


    detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_coord': ct_coord, 'ct_01': ct_01.float()}
    detection.update({'edge_hm': edge_hm})
    if 'fftcoef' in batch[0].keys():
        order = meta['fftorder'].max()
        fftcoef = torch.zeros([batch_size, ct_num, order*4], dtype=torch.float)
        fftloc = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
        if ct_num != 0:
            fftcoef[ct_01] = torch.Tensor(np.stack((sum([b['fftcoef'] for b in batch], [])),axis=0))
            fftloc[ct_01] = torch.Tensor(np.stack((sum([b['fftloc'] for b in batch], [])),axis=0))
        detection.update({'fftcoef':fftcoef, 'fftloc':fftloc})
    ret.update(detection)

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_4pys[ct_01] = torch.Tensor(np.stack((sum([b['i_it_4py'] for b in batch], [])),axis=0))
        c_it_4pys[ct_01] = torch.Tensor(np.stack((sum([b['c_it_4py'] for b in batch], [])),axis=0))
        i_gt_4pys[ct_01] = torch.Tensor(np.stack((sum([b['i_gt_4py'] for b in batch], [])),axis=0))
        c_gt_4pys[ct_01] = torch.Tensor(np.stack((sum([b['c_gt_4py'] for b in batch], [])),axis=0))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    # evolution
    i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_pys[ct_01] = torch.Tensor(np.stack((sum([b['i_it_py'] for b in batch], [])),axis=0))
        c_it_pys[ct_01] = torch.Tensor(np.stack((sum([b['c_it_py'] for b in batch], [])),axis=0))
        i_gt_pys[ct_01] = torch.Tensor(np.stack((sum([b['i_gt_py'] for b in batch], [])),axis=0))
        c_gt_pys[ct_01] = torch.Tensor(np.stack((sum([b['c_gt_py'] for b in batch], [])),axis=0))
    evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    ret.update(evolution)

    return ret



_collators = {
    'ssnake': ssnake_collator,
}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate

