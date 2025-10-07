from attrdict import AttrDict
import json


def read_config(path:str = './config.json'):
    # 读取JSON文件
    with open(path, 'r') as f:
        config_dict = json.load(f)

    # 将字典转换为具有属性访问的对象
    config = AttrDict(config_dict)

    return config