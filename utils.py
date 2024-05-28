import yaml
from types import SimpleNamespace as SN
import os
import shutil

def remove_reuslt_files(folder):
    shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_config():
    config_dir = '{0}'

    with open(config_dir.format('config.yaml'), "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    return SN(**config)


def get_test_object(args):
    if args.RESULT_SAVE == "neurons":
        test_object = args.HIDDEN_LAYER_NODES
    elif args.RESULT_SAVE == "batch_size":
        test_object = args.BATCH_SIZE
    else:
        test_object = args.TARGET_UPDATE

    return test_object


def get_refine_args(args, index):
    if args.RESULT_SAVE == "neurons":
        return args.HIDDEN_LAYER_NODES[index], args.BATCH_SIZE[2], args.TARGET_UPDATE[2]
    elif args.RESULT_SAVE == "batch_size":
        return args.HIDDEN_LAYER_NODES[2], args.BATCH_SIZE[index], args.TARGET_UPDATE[2]
    else:
        return args.HIDDEN_LAYER_NODES[2], args.BATCH_SIZE[2], args.TARGET_UPDATE[index]


def calc_exp_times(flow_time):
    min = int(round((flow_time % 3600) / 60, 0))
    sec = int(round((flow_time % 3600) % 60,0))

    str = "{0}분 {1}초".format(min, sec)

    return str

def print_exp_times(time_list, args):
    test_object = get_test_object(args)
    for i in range(len(time_list)):
        print("{0}_{1}의 소요시간 : {2}".format(test_object[i], args.RESULT_SAVE, time_list[i]))

