from lib.config import cfg, args
from lib.postprocessors import make_postprocessor
import os.path as osp

def visualize_vcoco(img_path,ckpt_path):
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network_given_ckpt
    import torch
    from lib.visualizers import make_visualizer
    network = make_network(cfg).cuda()
    load_network_given_ckpt(network, ckpt_path=ckpt_path)
    network.eval()
    visualizer = make_visualizer(cfg)
    postprocessor = make_postprocessor(cfg)
    # obtain batch for single image
    def obtain_batch_from_single_image(img_path):
        import cv2
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        from lib.utils.ssnake import snake_coco_utils
        inp, trans_in, trans_out, in_out_hw, orig_img, center, scale = \
            snake_coco_utils.augment_standard_only_img_inference(img)
        del img
        batch = {'inp': inp}
        img_name = osp.split(img_path)[-1]
        meta = {'img_name': img_name}
        batch.update({'meta':meta})
        batch = [batch]
        # collate
        from torch.utils.data.dataloader import default_collate
        ret = {'inp': default_collate([b['inp'] for b in batch])}
        meta = default_collate([b['meta'] for b in batch])
        ret.update({'meta': meta})
        batch = ret

        return batch
    batch = obtain_batch_from_single_image(img_path)

    for k in batch:
        if k != 'meta' and k != 'py':
            batch[k] = batch[k].cuda()
    with torch.no_grad():
        output = network(batch['inp'])
    if postprocessor is not None:
        postprocessor.post_processing(output)

        visualizer.visualize(output, batch)
    del network, visualizer, postprocessor, output
    # break

if __name__ == '__main__':
    img_path = args.img_path
    ckpt_path = args.ckpt_path
    if not osp.exists(img_path):
        print(f'Image does not exists in path {img_path}')
    else:
        visualize_vcoco(img_path,ckpt_path)
        print(f'Finish visualizing img {img_path}')
