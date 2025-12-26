import gc
import os
import time
import copy
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from torch.utils.tensorboard import SummaryWriter
from .utils import DisabledSummaryWriter, log_exceptions

if 'tensorboard' in configs['train'] and configs['train']['tensorboard'] and not configs['tune']['enable']:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/exp_{configs['model']['name']}_{configs['data']['name']}_layer_num{configs['model']['layer_num']}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(log_dir)
    #writer = SummaryWriter(log_dir='runs')
else:
    writer = DisabledSummaryWriter()


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

        log_dir = configs['train'].get('log_dir', None)
        self.writer = writer

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        self.writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

    @log_exceptions
    def train(self, model, tune_writer = False):
        self.create_optimizer(model)
        train_config = configs['train']

        if tune_writer:
            self.writer = SummaryWriter(log_dir=configs['train']['log_dir'])

        if not train_config['early_stop']:
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
            self.test(model)
            self.save_model(model)
            return model

        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # early stop
                    if now_patience == configs['train']['patience']:
                        break

            # re-initialize the model and load the best parameter
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
            for idx, k in enumerate(configs['test']['k']):
                self.writer.add_scalar(f'{configs['test']['metrics'][0]}@{k}/test', eval_result[configs['test']['metrics'][0]][idx], epoch_idx)
                self.writer.add_scalar(f'{configs['test']['metrics'][1]}@{k}/test', eval_result[configs['test']['metrics'][1]][idx], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            for idx, k in enumerate(configs['test']['k']):
                self.writer.add_scalar(f'{configs['test']['metrics'][0]}@{k}/test', eval_result[configs['test']['metrics'][0]][idx], epoch_idx)
                self.writer.add_scalar(f'{configs['test']['metrics'][1]}@{k}/test', eval_result[configs['test']['metrics'][1]][idx], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        else:
            raise NotImplemented
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            for idx, k in enumerate(configs['test']['k']):
                self.writer.add_scalar(f'{configs['test']['metrics'][0]}@{k}/test', eval_result[configs['test']['metrics'][0]][idx], configs['train']['epoch'])
                self.writer.add_scalar(f'{configs['test']['metrics'][1]}@{k}/test', eval_result[configs['test']['metrics'][1]][idx], configs['train']['epoch'])
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        else: 
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                timestamp = int(time.time())
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestamp))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
            return model
        else:
            raise KeyError("No pretrain_path in configs['train']")

"""
Special Trainer for General Collaborative Filtering methods (AutoCF, GFormer, ...)
"""
class AutoCFTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(AutoCFTrainer, self).__init__(data_handler, logger)
        self.fix_steps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            if i % self.fix_steps == 0:
                sampScores, seeds = model.sample_subgraphs()
                encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj)

            if i % self.fix_steps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
                loss_dict['infomax_loss'] = localGlobalLoss

            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        self.writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


class GFormerTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(GFormerTrainer, self).__init__(data_handler, logger)
        self.handler = data_handler
        self.user = configs['data']['user_num']
        self.item = configs['data']['item_num']
        self.latdim = configs['model']['embedding_size']
        self.fixSteps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        """ train in train mode """
        model.train()
        """ train Rec """
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        model.preSelect_anchor_set()
        # for recording loss
        loss_log_dict = {}
        # start this epoch
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            if i % self.fixSteps == 0:
                att_edge, add_adj = model.localGraph(self.handler.torch_adj, model.getEgoEmbeds(), self.handler)
                encoderAdj, decoderAdj, sub, cmp = model.masker(add_adj, att_edge)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj, sub, cmp)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)



"""
Special Trainer for Knowledge Graph-enhanced Recommendation methods (KGCL, ...)
"""
class KGCLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(KGCLTrainer, self).__init__(data_handler, logger)
        self.train_trans = configs['model']['train_trans']
        # if self.train_trans:
        #     self.triplet_dataloader = data_handler.triplet_dataloader

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.kgtrans_optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        """ train in train mode """
        model.train()
        """ train Rec """
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        # for recording loss
        loss_log_dict = {}
        # start this epoch
        kg_view_1, kg_view_2, ui_view_1, ui_view_2 = model.get_aug_views()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            batch_data.extend([kg_view_1, kg_view_2, ui_view_1, ui_view_2])
            loss, loss_dict = model.cal_loss(batch_data)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if self.train_trans:
            """ train KG trans """
            n_kg_batch = configs['data']['triplet_num'] // configs['train']['kg_batch_size']
            for iter in tqdm(range(n_kg_batch), desc='Training KG Trans', total=n_kg_batch):
                batch_data = self.data_handler.generate_kg_batch()
                batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))
                # feed batch_seqs into model.forward()
                kg_loss = model.cal_kg_loss(batch_data)

                self.kgtrans_optimizer.zero_grad(set_to_none=True)
                kg_loss.backward()
                self.kgtrans_optimizer.step()

                if 'kg_loss' not in loss_log_dict:
                    loss_log_dict['kg_loss'] = float(kg_loss) / n_kg_batch
                else:
                    loss_log_dict['kg_loss'] += float(kg_loss) / n_kg_batch

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)



class DiffKGTrainer(Trainer):
    def __init__(self, data_handler, logger):
        from models.kg.diffkg import GaussianDiffusion, Denoise
        super(DiffKGTrainer, self).__init__(data_handler, logger)
        self.diffusion = GaussianDiffusion(configs['model']['noise_scale'], configs['model']['noise_min'], configs['model']['noise_max'], configs['model']['steps']).cuda()
        out_dims = eval(configs['model']['dims']) + [configs['data']['entity_num']]
        in_dims = out_dims[::-1]
        self.denoise = Denoise(in_dims, out_dims, configs['model']['d_emb_size'], norm=True).cuda()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.optimizer_denoise = optim.Adam(self.denoise.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
    
    def train_epoch(self, model, epoch_idx):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        diffusionLoader = self.data_handler.diffusionLoader

        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        model.train()
        for _, tem in tqdm(enumerate(diffusionLoader), desc='Training Diffusion', total=len(diffusionLoader)):
            batch_data = list(map(lambda x: x.to(configs['device']), tem))

            ui_matrix = self.data_handler.ui_matrix
            iEmbeds = model.getEntityEmbeds().detach()
            uEmbeds = model.getUserEmbeds().detach()

            self.optimizer_denoise.zero_grad()
            loss_diff, loss_dict_diff = self.diffusion.cal_loss_diff(self.denoise, batch_data, ui_matrix, uEmbeds, iEmbeds)
            loss_diff.backward()
            self.optimizer_denoise.step()

        with torch.no_grad():
            denoised_edges = []
            h_list = []
            t_list = []

            for _, tem in enumerate(diffusionLoader):
                batch_data = list(map(lambda x: x.to(configs['device']), tem))
                batch_item, batch_index = batch_data
                denoised_batch = self.diffusion.p_sample(self.denoise, batch_item, configs['model']['sampling_steps'])
                top_item, indices_ = torch.topk(denoised_batch, k=configs['model']['rebuild_k'])
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        h_list.append(batch_index[i])
                        t_list.append(indices_[i][j])

            edge_set = set()
            for index in range(len(h_list)):
                edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
            for index in range(len(h_list)):
                if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
                    h_list.append(t_list[index])
                    t_list.append(h_list[index])
            
            relation_dict = self.data_handler.relation_dict
            for index in range(len(h_list)):
                try:
                    denoised_edges.append([h_list[index], t_list[index], relation_dict[int(h_list[index].cpu().numpy())][int(t_list[index].cpu().numpy())]])
                except Exception:
                    continue
            graph_tensor = torch.tensor(denoised_edges)
            index_ = graph_tensor[:, :-1]
            type_ = graph_tensor[:, -1]
            denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
            model.setDenoisedKG(denoisedKG)

        with torch.no_grad():
            index_, type_ = denoisedKG
            mask = ((torch.rand(type_.shape[0]) + configs['model']['keepRate']).floor()).type(torch.bool)
            denoisedKG = (index_[:, mask], type_[mask])
            self.generatedKG = denoisedKG
        
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            self.optimizer.zero_grad()

            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data, denoisedKG)
            ep_loss += loss.item()
            loss.backward()

            self.optimizer.step()

            loss_dict = {**loss_dict}
            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val
        
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

        
