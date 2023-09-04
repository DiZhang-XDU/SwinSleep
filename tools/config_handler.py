import yaml
from os.path import join

default_cfg = {
    
}

class yamlStruct:
    def __init__(self) -> None:
        pass
    def add(self, idx, elem):
        exec("self.%s = elem"%(idx))
    def load_dict(self, yaml_dict):
        for y in yaml_dict:
            if type(yaml_dict[y]) is not dict:
                self.add(y, yaml_dict[y])
            else:
                ys = yamlStruct()
                ys.load_dict(yaml_dict[y])
                self.add(y, ys)

class YamlHandler:
    def __init__(self, file) -> None:
        self.file = file
        
    def read_yaml(self, encoding = 'utf-8'):
        ys = yamlStruct()
        # ys.load_default()
        with open(self.file, 'r', encoding=encoding) as f:
            yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        ys.load_dict(yaml_dict)
        # get a certain experiment path
        if 'experiments' not in ys.path.weights:
            ys.path.weights = join('./experiments', ys.experiment, ys.path.weights)
            ys.path.tblogs = join('./experiments', ys.experiment, ys.path.tblogs)
            ys.path.log = join('./experiments', ys.experiment, ys.path.log)
        return ys

    def write_yaml(self, data, encoding = 'utf-8'):
        with open(self.file, 'w', encoding=encoding) as f:
            return yaml.dump(data, stream=f, allow_unicode=True)

if __name__ == '__main__':
    yh = YamlHandler('config\swin_shhs1_client.yaml')
    cfg = yh.read_yaml()
    print(cfg.scheduler)