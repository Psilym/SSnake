from lib.utils.ssnake import snake_config
from .ct_snake import get_network as get_ro


_network_factory = {
    'ro': get_ro
}


def get_network(cfg):
    arch = cfg.network
    heads = cfg.heads
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]
    if 'component' in list(cfg.keys()):
        cfg_component = cfg['component']
    else:
        cfg_component = None
    network = get_model(num_layers, heads, head_conv, snake_config.down_ratio, cfg.det_dir, cfg_component) # call .ct_snake/get_network function
    return network

