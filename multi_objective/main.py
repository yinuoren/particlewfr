import os
import sys
import torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import random
import numpy as np

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

import argparse
import os
import pathlib
import time
import json
from torch.utils import data
from tqdm import trange

import settings as s
import utils
from objectives import from_name
from scores import from_objectives
from hv import HyperVolume


from methods import HypernetMethod, ParetoMTLMethod, SingleTaskMethod, COSMOSMethod, MGDAMethod, UniformScalingMethod, ARGMOMethod, ParticleMethod


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def method_from_name(method, **kwargs):
    if method == 'ParetoMTL':
        return ParetoMTLMethod(**kwargs)
    elif 'cosmos' in method:
        return COSMOSMethod(**kwargs)
    elif method == 'SingleTask':
        return SingleTaskMethod(**kwargs)
    elif 'hyper' in method:
        return HypernetMethod(**kwargs)
    elif method == 'mgda':
        return MGDAMethod(**kwargs)
    elif method == 'uniform':
        return UniformScalingMethod(**kwargs)
    elif method == 'argmo':
        return ARGMOMethod(**kwargs)
    elif method == 'particle':
        return ParticleMethod(**kwargs)
    else:
        raise ValueError("Unkown method {}".format(method))


epoch_max = -1
volume_max = -1
elapsed_time = 0


def evaluate(j, e, method, scores, data_loader, logdir, reference_point, split, result_dict, clip, rank=None):
    assert split in ['train', 'val', 'test']
    global volume_max
    global epoch_max

    score_values = np.array([])
    for batch in data_loader:
        batch = utils.dict_to_cuda(batch, rank)
        if clip is not None:
            batch['clip'] = clip
        
        # more than one solution for some solvers
        s = []
        for l in method.eval_step(batch):
            batch.update(l)
            s.append([s(**batch) for s in scores])
        if score_values.size == 0:
            score_values = np.array(s)
        else:
            score_values += np.array(s)
        
    score_values /= len(data_loader)
    
    hv = HyperVolume(reference_point) 

    volume = hv.compute(score_values)
    # Computing hyper-volume for many objectives is expensive
    # volume = hv.compute(score_values) if score_values.shape[1] < 5 else -1

    if len(scores) == 2:
        pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir, "{}_{:03d}".format(split, e))
        pareto_front.append(score_values)
        pareto_front.plot()

    result = {
        "scores": score_values.tolist(),
        "hv": volume,
    }

    if split == 'train':
        if volume > volume_max:
            volume_max = volume
            epoch_max = e
                    
        result.update({
            "max_epoch_so_far": epoch_max,
            "max_volume_so_far": volume_max,
            "training_time_so_far": elapsed_time,
        })
    elif split == 'test':
        result.update({
            "training_time_so_far": elapsed_time,
        })

    result.update(method.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    
    return result_dict


def main(settings):
    print("start processing with settings", settings)
    utils.set_seed(settings['seed'])

    global elapsed_time

    # Create the experiment folders
    
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)



    ## Prepare the dataset
    
    train_set = utils.dataset_from_name(split='train', **settings)
    # val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'], collate_fn=train_set.collate_fn(**settings) if hasattr(train_set, 'collate_fn') else None)
    # val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True, num_workers=settings['num_workers'], collate_fn=val_set.collate_fn(**settings) if hasattr(val_set, 'collate_fn') else None)
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'], collate_fn=test_set.collate_fn(**settings) if hasattr(test_set, 'collate_fn') else None)



    ## Prepare the objectives, scores and method

    objectives = from_name(settings.pop('objectives'), train_set.task_names(), **settings)
    scores_train = from_objectives(objectives, train=True, **settings)
    if not settings['explicit']:
        scores_test = from_objectives(objectives, train=False, **settings)
    method = method_from_name(objectives=objectives, **settings)
    # rm1 = utils.RunningMean(400)
    # rm2 = utils.RunningMean(400)



    ## Run the experiment    

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    with open(pathlib.Path(logdir) / "settings.json", "w") as file:
        json.dump(train_results, file)

    for j in range(settings['num_starts']):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(method.model_params(), settings['lr'])
        
        if 'argmo' in settings['method']:
            optimizer_p = torch.optim.Adam([{'params': method.particles, 'lr': settings['p_lr']}])
            stage_count = 0
        
        if settings['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])
        
        epoch_iter = trange(settings['epochs'])
        for e in epoch_iter:
            tick = time.time()
            method.new_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to_cuda(batch)
                batch['lr'] = optimizer.param_groups[0]['lr'] 
                if settings['clip'] is not None:
                    batch['clip'] = settings['clip']
                
                if 'argmo' in settings['method']:
                    modular = (settings['n_particles']+settings['num_rv_adp'])
                    stage = 0 if stage_count % modular < settings['n_particles'] else 1
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    batch['stage'] = stage
                    batch['const'] = settings['const']
                    batch['pi'] = stage_count % modular
                    batch['use_p'] = e >= settings['warm']
                    batch['method'] = settings['rv_method']
                    
                    if stage == 0:
                        stats = method.step(batch)
                        optimizer.step()
                    else:
                        if batch['use_p']:
                            stats = method.step(batch)
                            optimizer_p.step()
                            
                    stage_count += 1
                else:
                    optimizer.zero_grad()
                    stats = method.step(batch)
                    optimizer.step()
                
                
                loss, _ = stats if isinstance(stats, tuple) else (stats, 0)
                epoch_iter.set_description(f"Epoch {e:03d}, batch: {b:03d}, train_loss {loss:.3f}, hv {volume_max:.3f}")
                
            tock = time.time()
            elapsed_time += tock - tick

            if settings['use_scheduler']:
                train_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
                scheduler.step()

            # Run eval on train set 
            
            if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
                train_results = evaluate(j, e, method, scores_train, train_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='train',
                    result_dict=train_results,
                    clip=settings['clip'] if settings['clip'] is not None else None)

            
            if settings['eval_every'] > 0 and (e+1) % settings['eval_every'] == 0 and not settings['explicit']:
                # Validation results
                # val_results = evaluate(j, e, method, scores_train, val_loader, logdir, 
                #     reference_point=settings['reference_point'],
                #     split='val',
                #     result_dict=val_results,
                    # clip=settings['clip'] if settings['clip'] is not None else None)

                # Test results
                
                test_results = evaluate(j, e, method, scores_test, test_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='test',
                    result_dict=test_results,
                    clip=settings['clip'] if settings['clip'] is not None else None)

            # Checkpoints
            
            if settings['checkpoint_every'] > 0 and (e+1) % settings['checkpoint_every'] == 0:
                pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        print("epoch_max={}, train_volume_max={}".format(epoch_max, volume_max))
        pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))
    return volume_max



def main_ddp(rank, world_size, settings):
    setup(rank, world_size)
    if rank in [-1, 0]:
        print("start processing with settings", settings)
    dist.barrier()
    print(f"start processing on rank {rank}")
    utils.set_seed(settings['seed'])

    global elapsed_time

    # Create the experiment folders
    if rank in [-1, 0]:
        logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    ## Prepare the dataset
    
    train_set = utils.dataset_from_name(split='train', **settings)
    # val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=False,sampler=DistributedSampler(train_set), 
                                   num_workers=settings['num_workers'], collate_fn=train_set.collate_fn(**settings) if hasattr(train_set, 'collate_fn') else None)
    # val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True, num_workers=settings['num_workers'], collate_fn=val_set.collate_fn(**settings) if hasattr(val_set, 'collate_fn') else None)
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'], collate_fn=test_set.collate_fn(**settings) if hasattr(test_set, 'collate_fn') else None)

    ## Prepare the objectives, scores and method

    objectives = from_name(settings.pop('objectives'), train_set.task_names(), **settings)
    scores_train = from_objectives(objectives, train=True, **settings)
    if not settings['explicit']:
        scores_test = from_objectives(objectives, train=False, **settings)
    method = method_from_name(objectives=objectives, **settings)
    method.model = DDP(method.model.to(rank), device_ids=[rank])
    # rm1 = utils.RunningMean(400)
    # rm2 = utils.RunningMean(400)

    ## Run the experiment    
    if rank in [-1, 0]:
        train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
        val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
        test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
        with open(pathlib.Path(logdir) / "settings.json", "w") as file:
            json.dump(train_results, file)

    for j in range(settings['num_starts']):

        if rank in [-1, 0]:
            train_results[f"start_{j}"] = {}
            val_results[f"start_{j}"] = {}
            test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(method.model_params(), settings['lr'])
        
        if 'argmo' in settings['method']:
            optimizer_p = torch.optim.Adam([{'params': method.particles, 'lr': settings['p_lr']}])
            stage_count = 0
        
        if settings['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])
        
        epoch_iter = trange(settings['epochs'])
        for e in epoch_iter:
            tick = time.time()
            method.new_epoch(e)
            train_loader.sampler.set_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to_cuda(batch, rank)
                batch['lr'] = optimizer.param_groups[0]['lr'] 
                if settings['clip'] is not None:
                    batch['clip'] = settings['clip']
                
                if 'argmo' in settings['method']:
                    modular = (settings['n_particles']+settings['num_rv_adp'])
                    stage = 0 if stage_count % modular < settings['n_particles'] else 1
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    batch['stage'] = stage
                    batch['const'] = settings['const']
                    batch['pi'] = stage_count % modular
                    batch['use_p'] = e >= settings['warm']
                    batch['method'] = settings['rv_method']
                    
                    if stage == 0:
                        stats = method.step(batch)
                        optimizer.step()
                    else:
                        if batch['use_p']:
                            stats = method.step(batch)
                            optimizer_p.step()
                            
                    stage_count += 1
                else:
                    optimizer.zero_grad()
                    stats = method.step(batch, rank)
                    optimizer.step()
                
                
                loss, _ = stats if isinstance(stats, tuple) else (stats, 0)
                epoch_iter.set_description(f"Epoch {e:03d}, batch: {b:03d}, train_loss {loss:.3f}, hv {volume_max:.3f}")


            tock = time.time()
            elapsed_time += tock - tick

            if settings['use_scheduler']:
                train_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
                scheduler.step()

            # Run eval on train set 
            
            if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0 and rank in [-1, 0]:
                train_results = evaluate(j, e, method, scores_train, train_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='train',
                    result_dict=train_results,
                    clip=settings['clip'] if settings['clip'] is not None else None,
                    rank=rank)

            
            if settings['eval_every'] > 0 and (e+1) % settings['eval_every'] == 0 and not settings['explicit'] and rank in [-1, 0]:
                # Validation results
                # val_results = evaluate(j, e, method, scores_train, val_loader, logdir, 
                #     reference_point=settings['reference_point'],
                #     split='val',
                #     result_dict=val_results,
                    # clip=settings['clip'] if settings['clip'] is not None else None)

                # Test results
                
                test_results = evaluate(j, e, method, scores_test, test_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='test',
                    result_dict=test_results,
                    clip=settings['clip'] if settings['clip'] is not None else None,
                    rank=rank)

            # Checkpoints
            
            if settings['checkpoint_every'] > 0 and (e+1) % settings['checkpoint_every'] == 0 and rank in [-1, 0]:
                pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        if rank in [-1, 0]:
            print("epoch_max={}, train_volume_max={}".format(epoch_max, volume_max))
            pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
            torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))
    
    cleanup()
    return volume_max

def run_ddp(demo_fn, world_size, settings):
    mp.spawn(demo_fn,
             args=(world_size, settings),
             nprocs=world_size,
             join=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='mm', help="The dataset to run on.")
    parser.add_argument('--method', '-m', default='cosmos', help="The method to generate the Pareto front.")
    parser.add_argument('--seed', '-s', default=1, type=int, help="Seed")
    parser.add_argument('--task_id', '-t', default=None, type=int, help='Task id to run single task in parallel. If not set then sequentially.')
    args = parser.parse_args()

    settings = s.generic
    
    if args.method == 'single_task':
        settings.update(s.SingleTaskSolver)
        if args.task_id is not None:
            settings['num_starts'] = 1
            settings['task_id'] = args.task_id
    elif args.method == 'cosmos':
        settings.update(s.cosmos)
    elif args.method == 'hyper_ln':
        settings.update(s.hyperSolver_ln)
    elif args.method == 'hyper_epo':
        settings.update(s.hyperSolver_epo)
    elif args.method == 'pmtl':
        settings.update(s.paretoMTL)
    elif args.method == 'mgda':
        settings.update(s.mgda)
    elif args.method == 'uniform':
        settings.update(s.uniform_scaling)
    elif args.method == 'argmo_kernel':
        settings.update(s.argmo_kernel)
    elif args.method == 'argmo_hv':
        settings.update(s.argmo_hv)
    elif args.method == 'particle':
        settings.update(s.particle)
    
    if args.dataset == 'mm':
        settings.update(s.multi_mnist)
    elif args.dataset == 'adult':
        settings.update(s.adult)
    elif args.dataset == 'mfm':
        settings.update(s.multi_fashion_mnist)
    elif args.dataset == 'fm':
        settings.update(s.multi_fashion)
    elif args.dataset == 'credit':
        settings.update(s.credit)
    elif args.dataset == 'compass':
        settings.update(s.compass)
    elif args.dataset == 'celeba':
        settings.update(s.celeba)
    elif args.dataset == 'mslr':
        settings.update(s.mslr)
    elif args.dataset == 'fonseca':
        settings.update(s.fonseca)
    elif args.dataset == 'zdt3':
        settings.update(s.ZDT3)
    
    settings['seed'] = args.seed

    return settings



if __name__ == "__main__":
    
    settings = parse_args()
    if settings['method'] == 'particle':
        run_ddp(main_ddp, world_size=4, settings=settings)
    else:
        main(settings)
