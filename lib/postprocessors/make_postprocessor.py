import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog


def _postprocessor_factory(cfg):
    task = cfg.task
    data_source = DatasetCatalog.get(cfg.test.dataset)['id']
    module = '.'.join(['lib.postprocessors', data_source, task])
    path = os.path.join('lib/postprocessors', data_source, task+'.py')
    postprocessor = imp.load_source(module, path).Postprocessor(cfg)
    return postprocessor


def make_postprocessor(cfg):
    return _postprocessor_factory(cfg)
