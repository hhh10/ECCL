import collections
import logging
import random
import time
import torch
# from datasets.build import build_filter_loader
from model import objectives
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F
from model.cm import ClusterMemory, InstanceMemory 
from collections import defaultdict
from tqdm import tqdm

def do_train(start_epoch, args, model, train_loader, evaluator0,evaluator1,evaluator2, optimizer,
             scheduler, checkpointer, trainset,num_classes,num_instance):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    if get_rank() == 0:
        logger.info("Validation before training - Epoch: {}".format(-1))
        top1_0 = evaluator0.eval(model.eval())
        top1_1 = evaluator1.eval(model.eval())
        top1_2 = evaluator2.eval(model.eval())
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "fushion_sdm_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "memory_loss_i2t": AverageMeter(),
        "memory_loss_t2i": AverageMeter(),
        "instance_memory_loss_i2t": AverageMeter(),
        "instance_memory_loss_t2i": AverageMeter(),
        "TAL_loss": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1_0 = 0.0
    best_top1_1 = 0.0
    best_top1_2 = 0.0
    if 'memory' in args.loss_names:
        with torch.no_grad():
            memory_features_image = defaultdict(list)
            memory_features_text = defaultdict(list)
            pid_container = set()
            
            for i, batch in tqdm(enumerate(train_loader)):
                # 将 batch 中的数据移动到设备上
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                images = batch['images']
                pids = batch['pids']
                captions = batch['caption_ids']
                
                # 模型仅输出 img_feats 和 text_feats，不再有 part_feats
                img_feats = model(img=images)
                text_feats = model(text=captions)
                
                # 将特征按照 pid 进行收集
                for img_feature, text_feature, pid in zip(img_feats, text_feats, pids):
                    pid_container.add(int(pid))
                    memory_features_image[int(pid)].append(img_feature)
                    memory_features_text[int(pid)].append(text_feature)
                    
            # 确定 Memory Bank 的大小和特征维度
            num_memory_features = len(pid_container)
            memory_features_dim = img_feats.shape[1]  # 特征的维度
            
            # 初始化 Memory Bank
            memory_bank_image = torch.zeros(num_classes, memory_features_dim).to(device)
            memory_bank_text = torch.zeros(num_classes, memory_features_dim).to(device)
            
            # 计算每个 pid 对应的特征平均值，存入 Memory Bank
            for pid in pid_container:
                image_feature_list = memory_features_image[pid]
                text_feature_list = memory_features_text[pid]
                
                image_feature = torch.stack(image_feature_list).mean(dim=0)
                text_feature = torch.stack(text_feature_list).mean(dim=0)
                
                memory_bank_image[pid] = image_feature
                memory_bank_text[pid] = text_feature
            
        # 使用 ClusterMemory 类进行定义
        memory_image = ClusterMemory(
            num_features=memory_features_dim,
            num_samples=num_classes,
            temp=args.temp,
            momentum=args.momentum,
            use_hard=args.use_hard,
            margin=args.margin
        ).to(device)
        
        memory_text = ClusterMemory(
            num_features=memory_features_dim,
            num_samples=num_classes,
            temp=args.temp,
            momentum=args.momentum,
            use_hard=args.use_hard,
            margin=args.margin
        ).to(device)
        
        # 将 Memory Bank 的特征进行归一化，并赋值给 ClusterMemory 的 features
        memory_image.features = F.normalize(memory_bank_image, dim=1)
        memory_text.features = F.normalize(memory_bank_text, dim=1)
        
    if 'instance_memory' in args.loss_names:
        with torch.no_grad():
            instance_memory_features_image = defaultdict(list)
            instance_memory_features_text = defaultdict(list)
            instance_id_container = set()
            memory_pids = torch.zeros(num_instance).to(device)
            
            for i, batch in tqdm(enumerate(train_loader)):
                # 将 batch 中的数据移动到设备上
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                images = batch['images']
                captions = batch['caption_ids']
                instance_ids = batch['image_ids']
                pids = batch['pids']
                
                # 模型仅输出 img_feats 和 text_feats，不再有 part_feats
                img_feats = model(img=images)
                text_feats = model(text=captions)
                
                # 将特征按照 instance_id 进行收集
                for img_feature, text_feature, instance_id, pid in zip(img_feats, text_feats, instance_ids, pids):
                    instance_id_container.add(int(instance_id))
                    instance_memory_features_image[int(instance_id)].append(img_feature)
                    instance_memory_features_text[int(instance_id)].append(text_feature)
                    memory_pids[instance_id] = pid
                    
            # 确定 Memory Bank 的大小和特征维度
            # num_memory_features = len(pid_container)
            memory_features_dim = img_feats.shape[1]  # 特征的维度
            
            # 初始化 Memory Bank
            memory_bank_image = torch.zeros(num_instance, memory_features_dim).to(device)
            memory_bank_text = torch.zeros(num_instance, memory_features_dim).to(device)
            
            # 计算每个 pid 对应的特征平均值，存入 Memory Bank
            for instance_id in instance_id_container:
                image_feature_list = instance_memory_features_image[instance_id]
                text_feature_list = instance_memory_features_text[instance_id]
                
                image_feature = torch.stack(image_feature_list).mean(dim=0)
                text_feature = torch.stack(text_feature_list).mean(dim=0)
                
                memory_bank_image[instance_id] = image_feature
                memory_bank_text[instance_id] = text_feature
            
            
        # 使用 ClusterMemory 类进行定义
        # instance_memory_image = ClusterMemory(
        #     num_features=memory_features_dim,
        #     num_samples=num_instance,
        #     temp=args.temp,
        #     momentum=args.momentum,
        #     use_hard=args.use_hard,
        #     margin=args.margin
        # ).to(device)
        
        # instance_memory_text = ClusterMemory(
        #     num_features=memory_features_dim,
        #     num_samples=num_instance,
        #     temp=args.temp,
        #     momentum=args.momentum,
        #     use_hard=args.use_hard,
        #     margin=args.margin
        # ).to(device)
        instance_memory_image = InstanceMemory(
            num_features=memory_features_dim,
            num_instances=num_instance,
            temp=args.temp_instance,
            momentum=args.momentum,
            margin=args.margin
        ).to(device)
        
        instance_memory_text = InstanceMemory(
            num_features=memory_features_dim,
            num_instances=num_instance,
            temp=args.temp_instance,
            momentum=args.momentum,
            margin=args.margin
        ).to(device)
        
        # 将 Memory Bank 的特征进行归一化，并赋值给 ClusterMemory 的 features
        instance_memory_image.features = F.normalize(memory_bank_image, dim=1)
        instance_memory_text.features = F.normalize(memory_bank_text, dim=1)    
        instance_memory_image.pids = memory_pids
        instance_memory_text.pids = memory_pids
        
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            if 'memory' in args.loss_names and 'instance_memory' in args.loss_names:
                ret = model(batch=batch,img_memory=memory_image,text_memory=memory_text,
                            instance_img_memory=instance_memory_image,instance_text_memory=instance_memory_text)
            elif 'memory' in args.loss_names:
                ret = model(batch=batch,img_memory=memory_image,text_memory=memory_text)
            
            
            # ret = {key: values.mean() for key, values in ret.items()}
            else:
                ret = model(batch=batch)
            # ret = model(batch)
            ret = {key: values.mean() for key, values in ret.items()}
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['fushion_sdm_loss'].update(ret.get('fushion_sdm_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['memory_loss_i2t'].update(ret.get('memory_loss_i2t', 0), batch_size)
            meters['memory_loss_t2i'].update(ret.get('memory_loss_t2i', 0), batch_size)
            meters['instance_memory_loss_i2t'].update(ret.get('instance_memory_loss_i2t', 0), batch_size)
            meters['instance_memory_loss_t2i'].update(ret.get('instance_memory_loss_t2i', 0), batch_size)
            meters['TAL_loss'].update(ret.get('TAL_loss', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / 60
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[min] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            logger.info(f"best R1: CUHK {best_top1_0}, ICFG {best_top1_1}, RSTP {best_top1_2}")
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1_0 = evaluator0.eval(model.module.eval())
                    top1_1 = evaluator1.eval(model.module.eval())
                    top1_2 = evaluator2.eval(model.module.eval())
                else:
                    top1_0 = evaluator0.eval(model.eval())
                    top1_1 = evaluator1.eval(model.eval())
                    top1_2 = evaluator2.eval(model.eval())
                torch.cuda.empty_cache()
                if best_top1_0 < top1_0:
                    best_top1_0 = top1_0
                    arguments["epoch"] = epoch
                    checkpointer.save("best0", **arguments)
                if best_top1_1 < top1_1:
                    best_top1_1 = top1_1
                    arguments["epoch"] = epoch
                    checkpointer.save("best1", **arguments)
                if best_top1_2 < top1_2:
                    best_top1_2 = top1_2
                    arguments["epoch"] = epoch
                    checkpointer.save("best2", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1_0}, {best_top1_1}, {best_top1_2} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())