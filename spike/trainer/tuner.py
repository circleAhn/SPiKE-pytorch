from models.bulid_model import build_model
from config.configurator import configs
import torch
from trainer.trainer import init_seed
import datetime
from torch.utils.tensorboard import SummaryWriter

class Tuner(object):
    def __init__(self, logger):
        self.logger = logger
        self.hyperparameters = configs['tune']['hyperparameters']
        self.tune_list = []
        self.search_length = 1
        for hyper_para in self.hyperparameters:
            self.tune_list.append(configs['tune'][hyper_para])
            self.search_length = self.search_length * len(configs['tune'][hyper_para])
        self.para_length = [len(para_list) for para_list in self.tune_list]
        self.hex_length = [1 for _ in range(len(self.tune_list))]
        for i in range(len(self.para_length) - 2, -1, -1):
            self.hex_length[i] = self.para_length[i + 1] * self.hex_length[i + 1]
        self.origin_model_para = configs['model'].copy()

    def zero_step(self):
        self.now_step = 0

    def step(self):
        self.now_step += 1

    def next_model(self, data_handler):
        init_seed()
        now_para = {}
        now_para_str = ''
        for i in range(len(self.hyperparameters)):
            para_name = self.hyperparameters[i]
            selected_idx = (self.now_step // self.hex_length[i]) % self.para_length[i]
            seleted_val = self.tune_list[i][selected_idx]
            now_para[para_name] = seleted_val
            now_para_str += '{}{}'.format(para_name, seleted_val)
            configs['model'][para_name] = seleted_val
        configs['tune']['now_para_str'] = now_para_str
        self.logger.log('hyperparameter: {}'.format(now_para))
        model = build_model(data_handler).cuda()

        log_dir = self.generate_log_dir(configs['model']['name'], configs['data']['name'], now_para_str)
        configs['train']['log_dir'] = log_dir
        return model

    def grid_search(self, data_handler, trainer):
        self.zero_step()
        for _ in range(self.search_length):
            model = self.next_model(data_handler)
            trainer.train(model, tune_writer = True)
            # trainer.evaluate(model)
            del model
            torch.cuda.empty_cache()
            self.step()
        configs['model'] = self.origin_model_para.copy()

    def generate_log_dir(self, model_name, data_name, now_para_str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"runs/exp_{model_name}_{data_name}_{now_para_str}_{timestamp}"
