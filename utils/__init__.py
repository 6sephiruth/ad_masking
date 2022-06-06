from utils.adv_util import *
from utils.data_util import *
from utils.mask_util import *
from utils.util import *

import argparse, yaml

class Args(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_params():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('params', type=str)
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params = yaml.safe_load(f)

    # initialize arguments with parameters
    ret = Args(**params)

    # designate gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(ret.gpu)

    # set seed
    np.random.seed(ret.seed)
    torch.manual_seed(ret.seed)

    # set dataset dir
    if 'data_dir' not in params.keys():
        ret.data_dir = './dataset'

    # set model dir
    ret.model_dir = "experiments/{}/{}_{}.pt".format(
                                ret.dataset,
                                ret.dataset,
                                ret.model_name
                            )

    # directories for experiments
    dirs = [ret.data_dir, f"./experiments/{ret.dataset}"]
    makedirs(dirs)

    # set cuda settings
    use_cuda = torch.cuda.is_available()
    ret.device = torch.device("cuda" if use_cuda else "cpu")

    # set train/test keyword arguments
    ret.kwargs = {'batch_size': ret.bat_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True}
        ret.kwargs.update(cuda_kwargs)

    return ret