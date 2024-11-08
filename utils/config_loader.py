import yaml
import argparse
from argparse import Namespace


def dict_to_namespace(namespace, dic):
    for k, v in dic.items():
        if isinstance(v, dict):
            sub_namespace = Namespace()
            setattr(namespace, k, sub_namespace)
            dict_to_namespace(sub_namespace, v)
        else:
            setattr(namespace, k, v)


def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.FullLoader)
        dict_to_namespace(args, dic)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, required=True, help="Please provide config file (.yaml).")
    args = parser.parse_args()
    assert args.c != "", "Please provide config file (.yaml). By using --c"
    load_yaml(args, args.c)
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)