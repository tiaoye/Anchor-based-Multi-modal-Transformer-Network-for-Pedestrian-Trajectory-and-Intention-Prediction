import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.utils.visualization import Visualizer
from model.utils.box_utils import cxcywh_to_x1y1x2y2
from model.utils.dataset_utils import restore
from model.modeling.gmm2d import GMM2D
from model.modeling.gmm4d import GMM4D
from .evaluate import evaluate_multimodal, compute_kde_nll, evaluate_intent
from .utils import print_info, viz_results, post_process

from tqdm import tqdm
import pickle as pkl
import pdb

def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    max_iters = len(dataloader)
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')


    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader,leave=False), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            intent_label = batch['intent_label'].to(device) if 'intent_label' in batch else None
            pred_intention = None
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            if not ('Gtppo' in cfg.METHOD):
                pred_goal, pred_traj, loss_dict, dist_goal, dist_traj = model(input_x, 
                                                                    y_global, 
                                                                    neighbors_st=neighbors_st, 
                                                                    adjacency=adjacency, 
                                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                    first_history_indices=first_history_indices)
            else: # new added
                pred_goal, pred_traj, loss_dict, dist_goal, dist_traj, pred_intention = model(input_x, 
                                                                    y_global, 
                                                                    neighbors_st=neighbors_st, 
                                                                    adjacency=adjacency, 
                                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                    first_history_indices=first_history_indices,
                                                                    intent_label=intent_label)
            if cfg.MODEL.LATENT_DIST == 'categorical':
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld'] - \
                       1. * loss_dict['mutual_info_p']
            else:
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld']
            if 'Gtppo' in cfg.METHOD:
                if cfg.MODEL.PRED_INTENTION is True:
                    loss += loss_dict['loss_intention']
            model.param_scheduler.step()
            loss_dict = {k:v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(epoch, model, optimizer, loss_dict, logger)

            # TODO: add vis for intention
            if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0:
                ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_goal=dist_goal)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
                intent_label = intent_label.detach().to('cpu').numpy() if intent_label is not None else None
                pred_intention = pred_intention.detach().to('cpu').numpy() if pred_intention is not None else None
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_train',intent_label=intent_label, pred_intention=pred_intention)
                
def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_goal_val = 0.0
    loss_traj_val = 0.0
    loss_KLD_val = 0.0
    loss_intention_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            intent_label = batch['intent_label'].to(device) if 'intent_label' in batch else None
            pred_intention = None
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            if not ('Gtppo' in cfg.METHOD):
                pred_goal, pred_traj, loss_dict, _, _ = model(input_x, 
                                                                y_global, 
                                                                neighbors_st=neighbors_st,
                                                                adjacency=adjacency,
                                                                cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                first_history_indices=first_history_indices)
            else: # new added
                pred_goal, pred_traj, loss_dict, _, _, pred_intention = model(input_x, 
                                                                y_global, 
                                                                neighbors_st=neighbors_st,
                                                                adjacency=adjacency,
                                                                cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                first_history_indices=first_history_indices,
                                                                intent_label=intent_label)
            # compute loss
            loss = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld']
            loss_goal_val += loss_dict['loss_goal'].item()
            loss_traj_val += loss_dict['loss_traj'].item()
            loss_KLD_val += loss_dict['loss_kld'].item()
            loss_intention_val += loss_dict['loss_intention'].item() if 'loss_intention' in loss_dict else 0.0
    loss_goal_val /= (iters + 1)
    loss_traj_val /= (iters + 1)
    loss_KLD_val /= (iters + 1)
    loss_intention_val /= (iters + 1)
    loss_val = loss_goal_val + loss_traj_val + loss_KLD_val + loss_intention_val
    
    info = "loss_val:{:.4f}, \
            loss_goal_val:{:.4f}, \
            loss_traj_val:{:.4f}, \
            loss_kld_val:{:.4f}".format(loss_val, loss_goal_val, loss_traj_val, loss_KLD_val)
    if 'loss_intention' in loss_dict:
        info += ", loss_intention_val:{:.4f}".format(loss_intention_val)
    if hasattr(logger, 'log_values'):
        logger.info(info)
        log_values_dict = {'loss_val':loss_val, 
                           'loss_goal_val':loss_goal_val,
                           'loss_traj_val':loss_traj_val, 
                           'loss_kld_val':loss_KLD_val}
        if 'loss_intention' in loss_dict:
            log_values_dict['loss_intention_val']=loss_intention_val
        logger.log_values(log_values_dict)#, step=epoch)

    else:
        print(info)
    return loss_val

def inference(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False):
    model.eval()
    all_img_paths = []
    all_X_globals = []
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_distributions = []
    all_timesteps = []
    all_pred_intentions = []
    all_intent_labels = []
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')
    
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            intent_label = batch['intent_label'].to(device) if 'intent_label' in batch else None
            pred_intention = None
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            if not ('Gtppo' in cfg.METHOD):
                pred_goal, pred_traj, _, dist_goal, dist_traj = model(input_x, 
                                                                    neighbors_st=neighbors_st,
                                                                    adjacency=adjacency,
                                                                    z_mode=False, 
                                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                    first_history_indices=first_history_indices)
            else: # new added
                pred_goal, pred_traj, _, dist_goal, dist_traj, pred_intention = model(input_x, 
                                                                    neighbors_st=neighbors_st,
                                                                    adjacency=adjacency,
                                                                    z_mode=False, 
                                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                    first_history_indices=first_history_indices,
                                                                    intent_label = intent_label)
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=dist_traj, dist_goal=dist_goal)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            intent_label = intent_label.detach().to('cpu').numpy() if intent_label is not None else None
            pred_intention = pred_intention.detach().to('cpu').numpy() if pred_intention is not None else None
            all_img_paths.extend(img_path)
            all_X_globals.append(X_global)
            all_pred_goals.append(pred_goal)
            all_pred_trajs.append(pred_traj)
            all_gt_goals.append(y_global[:, -1])
            all_gt_trajs.append(y_global)
            all_timesteps.append(batch['timestep'].numpy())
            all_pred_intentions.append(pred_intention)
            all_intent_labels.append(intent_label)
            if dist_traj is not None:
                all_distributions.append(dist_traj)
            else:
                all_distributions.append(dist_goal)
            # if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0:
            if cfg.VISUALIZE:
                for id_to_show in range(0,len(pred_traj),100):
                    l = [(12,400),(14,100),(8,100),(12,100)]
                    if (iters,id_to_show) not in l:
                        continue
                    name = 'pred_test_{}_{}'.format(iters, id_to_show)
                    viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj, 
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name=name,intent_label=intent_label, pred_intention=pred_intention,id_to_show=id_to_show) 
        
        # Evaluate
        all_X_globals = np.concatenate(all_X_globals, axis=0)
        all_pred_goals = np.concatenate(all_pred_goals, axis=0) if all_pred_goals[0] is not None else None
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
        all_gt_goals = np.concatenate(all_gt_goals, axis=0)
        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
        all_timesteps = np.concatenate(all_timesteps, axis=0)
        all_pred_intentions = np.concatenate(all_pred_intentions, axis=0) if all_pred_intentions[0] is not None else None
        all_intent_labels = np.concatenate(all_intent_labels, axis=0) if all_intent_labels[0] is not None else None
        if hasattr(all_distributions[0], 'mus'):
            distribution = model.GMM(torch.cat([d.input_log_pis for d in all_distributions], axis=0),
                                    torch.cat([d.mus for d in all_distributions], axis=0),
                                    torch.cat([d.log_sigmas for d in all_distributions], axis=0),
                                    torch.cat([d.corrs for d in all_distributions], axis=0))
        else:
            distribution = None 
        # eval_pred_results = evaluate(all_pred_goals, all_gt_goals)
        mode = 'bbox' if all_gt_trajs.shape[-1] == 4 else 'point'
        eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=distribution, bbox_type=cfg.DATASET.BBOX_TYPE)
        if all_pred_intentions is not None and all_intent_labels is not None:
            eval_results.update(evaluate_intent(all_pred_intentions, all_intent_labels))
        for key, value in eval_results.items():
            info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
            if hasattr(logger, 'log_values'):
                logger.info(info)
            else:
                print(info)
        
        if hasattr(logger, 'log_values'):
            logger.log_values(eval_results)

        if test_mode:
            # save inputs, redictions and targets for test mode
            outputs = {'img_path': all_img_paths, 'X_global': all_X_globals, 'timestep': all_timesteps,
                       'pred_trajs': all_pred_trajs, 'gt_trajs':all_gt_trajs,'distributions':distribution}
            if all_pred_intentions is not None and all_intent_labels is not None:
                outputs.update({'pred_intentions': all_pred_intentions, 'intent_labels': all_intent_labels})
            if not os.path.exists(cfg.OUT_DIR):
                os.makedirs(cfg.OUT_DIR)
            output_file = os.path.join(cfg.OUT_DIR, '{}_{}.pkl'.format(cfg.MODEL.LATENT_DIST, cfg.DATASET.NAME))
            print("Writing outputs to: ", output_file)
            pkl.dump(outputs, open(output_file,'wb'))

    # Mevaluate KDE NLL, since we sample 2000, need to use a smaller batchsize
    if eval_kde_nll:
        dataloader_params ={
            "batch_size": cfg.TEST.KDE_BATCH_SIZE,
            "shuffle": False,
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "collate_fn": dataloader.collate_fn,
            }
        kde_nll_dataloader = DataLoader(dataloader.dataset, **dataloader_params)
        # inference_kde_nll(cfg, epoch, model, kde_nll_dataloader, device, logger)

def inference_kde_nll(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_kde_nll = []
    all_per_step_kde_nll = []
    num_samples = model.K
    model.K = 2000
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            intent_label = batch['intent_label'].to(device) if 'intent_label' in batch else None
            pred_intention = None
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            if not ('Gtppo' in cfg.METHOD):
                pred_goal, pred_traj, _, _, _ = model(input_x, 
                                                        neighbors_st=neighbors_st,
                                                        adjacency=adjacency,
                                                        z_mode=False, 
                                                        cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                        first_history_indices=first_history_indices)
            else:
                pred_goal, pred_traj, _, _, _, _ = model(input_x, 
                                                        neighbors_st=neighbors_st,
                                                        adjacency=adjacency,
                                                        z_mode=False, 
                                                        cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                        first_history_indices=first_history_indices,
                                                        intent_label=intent_label)
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=None, dist_goal=None)
            intent_label = intent_label.detach().to('cpu').numpy() if intent_label is not None else None
            pred_intention = pred_intention.detach().to('cpu').numpy() if pred_intention is not None else None
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            for i in range(len(pred_traj)):
                KDE_NLL, KDE_NLL_PER_STEP = compute_kde_nll(pred_traj[i:i+1], y_global[i:i+1])
                all_kde_nll.append(KDE_NLL)
                all_per_step_kde_nll.append(KDE_NLL_PER_STEP)
        KDE_NLL = np.array(all_kde_nll).mean()
        KDE_NLL_PER_STEP = np.stack(all_per_step_kde_nll, axis=0).mean(axis=0)
        # Evaluate
        Goal_NLL = KDE_NLL_PER_STEP[-1]
        nll_dict = {'KDE_NLL': KDE_NLL} if cfg.MODEL.LATENT_DIST == 'categorical' else {'KDE_NLL': KDE_NLL, 'Goal_NLL': Goal_NLL}
        info = "Testing prediction KDE_NLL:{:.4f}, per step NLL:{}".format(KDE_NLL, KDE_NLL_PER_STEP)
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        if hasattr(logger, 'log_values'):
            logger.log_values(nll_dict)

    # reset model.K back to 20
    model.K = num_samples
    return KDE_NLL