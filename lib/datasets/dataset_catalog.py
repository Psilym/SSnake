from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'vcocoTrain': {
            'id': 'vcoco',
            'data_root': 'data/vcoco/train2017',
            'ann_file': 'data/vcoco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'vcocoVal': {
            'id': 'vcoco',
            'data_root': 'data/vcoco/val2017',
            'ann_file': 'data/vcoco/annotations/instances_val2017.json',
            'split': 'test'
        },
        'vcocoTest': {
            'id': 'vcoco_test',
            'data_root': 'data/vcoco/test2017',
            'ann_file': 'data/vcoco/annotations/instances_test2017.json',
            'split': 'test'
        },
    }
    # for cross fold
    num_fold=4
    for i_fold in range(num_fold):
        id = 'vcoco'
        data_root = f'data/vcoco/fold{i_fold}/train2017'
        ann_file = f'data/vcoco/fold{i_fold}/annotations/instances_train2017.json'
        split='train'
        dataset_attrs[f"vcocoTrain_fold{i_fold}"]={'id':id,'data_root':data_root,'ann_file':ann_file,'split':split}

        id = 'vcoco'
        data_root = f'data/vcoco/fold{i_fold}/val2017'
        ann_file = f'data/vcoco/fold{i_fold}/annotations/instances_val2017.json'
        split = 'val'
        dataset_attrs[f"vcocoVal_fold{i_fold}"] = {'id': id, 'data_root': data_root, 'ann_file': ann_file, 'split': split}

    print("add all fold catalog.")

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

