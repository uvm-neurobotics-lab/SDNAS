##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import os, sys, time, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, bilevel):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_inputs = base_inputs.cuda()
    base_targets = base_targets.cuda(non_blocking=True)
    arch_inputs = arch_inputs.cuda()
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    if not bilevel:
      a_optimizer.zero_grad()
    _, logits, cost = network(base_inputs)
    base_loss = criterion(logits, base_targets) + (cost/1e9)
    base_loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    if not bilevel:
      a_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    try: 
      base_losses.update(base_loss.item() - (cost.item()/1e9),  base_inputs.size(0))
    except:
      base_losses.update(base_loss.item() - (cost/1e9),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    if bilevel:
      # update the architecture-weight
      a_optimizer.zero_grad()
      _, logits, cost = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets) + (cost/1e9)
      arch_loss.backward()
      a_optimizer.step()
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
    else:
      arch_losses.update(0,  arch_inputs.size(0))
      arch_top1.update  (0, arch_inputs.size(0))
      arch_top5.update  (0, arch_inputs.size(0))


    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg

def train_func(xloader, network, criterion, scheduler, w_optimizer, epoch_str, print_freq, archs, arch_iter, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  val_losses, val_top1, val_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  for step, (base_inputs, base_targets, val_inputs, val_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    try:
      arch = next(arch_iter)
    except:
      arch_iter = iter(archs)
      arch = next(arch_iter)
    base_inputs = base_inputs.cuda()
    base_targets = base_targets.cuda(non_blocking=True)
    val_inputs = val_inputs.cuda()
    val_targets = val_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    _, logits, _ = network(base_inputs) #, arch)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # validate arch
    _, logits, _ = network(val_inputs, arch)
    val_loss = criterion(logits, val_targets)
    # record
    val_prec1, val_prec5 = obtain_accuracy(logits.data, val_targets.data, topk=(1, 5))
    val_losses.update(val_loss.item(),  val_inputs.size(0))
    val_top1.update  (val_prec1.item(), val_inputs.size(0))
    val_top5.update  (val_prec5.item(), val_inputs.size(0))
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*TRAIN* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Val  [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=val_losses, top1=val_top1, top5=val_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, val_losses.avg, val_top1.avg, val_top5.avg, arch_iter


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  #config_path = 'configs/nas-benchmark/algos/GDAS.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None and not args.constrain:
    model_config = dict2config({'name': 'SNAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'inp_size' : 0,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  elif xargs.model_config is None:
    model_config = dict2config({'name': 'SNAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'inp_size' : 32,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  search_model = get_cell_based_tiny_net(model_config)
  #logger.log('search-model :\n{:}'.format(search_model))
  logger.log('model-config : {:}'.format(model_config))
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  #logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  #network, criterion = search_model.cuda(), criterion.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: search_model.genotype()}

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  sampled_weights = []
  for epoch in range(start_epoch, total_epoch + config.t_epochs):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch + config.t_epochs), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    search_model.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
    logger.log('\n[Search the {:}-th epoch] {:}, tau={:}, LR={:}'.format(epoch_str, need_time, search_model.get_tau(), min(w_scheduler.get_lr())))
    if epoch < total_epoch:
      search_w_loss, search_w_top1, search_w_top5, valid_a_loss , valid_a_top1 , valid_a_top5 \
                = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger, xargs.bilevel)
    else:
      try:
        search_w_loss, search_w_top1, search_w_top5, valid_a_loss , valid_a_top1 , valid_a_top5, arch_iter \
                  = train_func(search_loader, network, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, sampled_weights[0], arch_iter, logger)
      except IndexError:
        weights = search_model.sample_weights(100)
        sampled_weights.append(weights)
        arch_iter = iter(weights)
        search_w_loss, search_w_top1, search_w_top5, valid_a_loss , valid_a_top1 , valid_a_top5, arch_iter \
                  = train_func(search_loader, network, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, sampled_weights[0], arch_iter, logger)

    search_time.update(time.time() - start_time)
    logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss , valid_a_top1 , valid_a_top5 ))

    if (epoch+1) % 50 == 0 and not config.t_epochs:
        weights = search_model.sample_weights(100)
        sampled_weights.append(weights)
    elif (epoch+1) == total_epoch and config.t_epochs:
        weights = search_model.sample_weights(100)
        sampled_weights.append(weights)
        arch_iter = iter(weights)
    # validate with single arch 
    single_weight = search_model.sample_weights(1)[0]
    single_valid_acc = AverageMeter()
    network.eval()
    for i in range(10):
      try:
        val_input, val_target = next(valid_iter)
      except Exception as e:
        valid_iter = iter(valid_loader)
        val_input, val_target = next(valid_iter)
      n_val = val_input.size(0)
      with torch.no_grad():
        val_target = val_target.cuda(non_blocking=True)
        _, logits, _ = network(val_input, weights=single_weight)
        val_acc1, val_acc5 = obtain_accuracy(logits.data, val_target.data, topk=(1,5))
        single_valid_acc.update(val_acc1.item(), n_val)
    logger.log('[{:}] valid : accuracy = {:.2f}'.format(epoch_str, single_valid_acc.avg))

    # check the best accuracy
    valid_accuracies[epoch] = valid_a_top1
    if valid_a_top1 > valid_accuracies['best']:
      valid_accuracies['best'] = valid_a_top1
      genotypes['best']        = search_model.genotype()
      find_best = True
    else: find_best = False
 
    if epoch < total_epoch:
      genotypes[epoch] = search_model.genotype()
      logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, valid_a_top1))
      copy_checkpoint(model_base_path, model_best_path, logger)
    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
    if api is not None and epoch < total_epoch: logger.log('{:}'.format(api.query_by_arch( genotypes[epoch] )))
    
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  network.eval()
  # Evaluate the architectures sampled throughout the search
  for i in range(len(sampled_weights)-1):
    logger.log('Sample eval : epoch {}'.format((i+1)*50-1))
    for w in sampled_weights[i]:
      sample_valid_acc = AverageMeter()
      for i in range(10):
        try:
          val_input, val_target = next(valid_iter)
        except Exception as e:
          valid_iter = iter(valid_loader)
          val_input, val_target = next(valid_iter)
        n_val = val_input.size(0)
        with torch.no_grad():
          val_target = val_target.cuda(non_blocking=True)
          _, logits, _ = network(val_input, weights=w)
          val_acc1, val_acc5 = obtain_accuracy(logits.data, val_target.data, topk=(1,5))
          sample_valid_acc.update(val_acc1.item(), n_val)
      w_gene = search_model.genotype(w)
      if api is not None: 
        ind = api.query_index_by_arch(w_gene)
        info = api.query_meta_info_by_index(ind)
        metrics = info.get_metrics('cifar10', 'ori-test')
        acc = metrics['accuracy']
      else: acc = 0.0
      logger.log('sample valid : val_acc = {:.2f} test_acc = {:.2f}'.format(sample_valid_acc.avg, acc))
  # Evaluate the final sampling separately to find the top 10 architectures 
  logger.log('Final sample eval')
  final_archs = []
  for w in sampled_weights[-1]:
    sample_valid_acc = AverageMeter()
    for i in range(10):
      try:
        val_input, val_target = next(valid_iter)
      except Exception as e:
        valid_iter = iter(valid_loader)
        val_input, val_target = next(valid_iter)
      n_val = val_input.size(0)
      with torch.no_grad():
        val_target = val_target.cuda(non_blocking=True)
        _, logits, _ = network(val_input, weights=w)
        val_acc1, val_acc5 = obtain_accuracy(logits.data, val_target.data, topk=(1,5))
        sample_valid_acc.update(val_acc1.item(), n_val)
    w_gene = search_model.genotype(w)
    if api is not None: 
      ind = api.query_index_by_arch(w_gene)
      info = api.query_meta_info_by_index(ind)
      metrics = info.get_metrics('cifar10', 'ori-test')
      acc = metrics['accuracy']
    else: acc = 0.0
    logger.log('sample valid : val_acc = {:.2f} test_acc = {:.2f}'.format(sample_valid_acc.avg, acc))
    final_archs.append((w, sample_valid_acc.avg))
  top_10 = sorted(final_archs, key=lambda x:x[1], reverse=True)[:10]
  # Evaluate the top 10 architectures on the entire validation set
  logger.log('Evaluating top archs')
  for w, prev_acc in top_10:
    full_valid_acc = AverageMeter()
    for val_input, val_target in valid_loader: 
      n_val = val_input.size(0)
      with torch.no_grad():
        val_target = val_target.cuda(non_blocking=True)
        _, logits, _ = network(val_input, weights=w)
        val_acc1, val_acc5 = obtain_accuracy(logits.data, val_target.data, topk=(1,5))
        full_valid_acc.update(val_acc1.item(), n_val)
    w_gene = search_model.genotype(w)
    logger.log('genotype {}'.format(w_gene))
    if api is not None: 
      ind = api.query_index_by_arch(w_gene)
      info = api.query_meta_info_by_index(ind)
      metrics = info.get_metrics('cifar10', 'ori-test')
      acc = metrics['accuracy']
    else: acc = 0.0
    logger.log('full valid : val_acc = {:.2f} test_acc = {:.2f} pval_acc = {:.2f}'.format(full_valid_acc.avg, acc, prev_acc))

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  logger.log('SNAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1]))
  if api is not None: logger.log('{:}'.format( api.query_by_arch(genotypes[total_epoch-1]) ))
  logger.close()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("SNAS")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--tau_min',            type=float,               help='The minimum tau for Gumbel')
  parser.add_argument('--tau_max',            type=float,               help='The maximum tau for Gumbel')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  # control different experiments
  parser.add_argument('--bilevel',            type=int,   choices=[0,1], help='Whether to train arch weights separately on valid data')
  parser.add_argument('--constrain',          type=int,   choices=[0,1], help='Whether to use resource constraint regularization')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
