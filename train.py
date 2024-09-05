import os
import time
import torch
import numpy
from eval import Evaluator
from model.build import build_yolo
from config import parse_args
from dataset.voc import VOCDataset
from dataset.utils import CollateFunc
from dataset.augment import Augmentation
from torch.utils.tensorboard import SummaryWriter

from utils.flops import flops
from utils.criterion import Loss
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lambda_lr_scheduler

def train():
    parser, args = parse_args()
    writer = SummaryWriter('log')
    print("Setting Arguments.. : ")
    for action in parser._actions:
        if action.dest != 'help':
            print(f"{action.dest} = {getattr(args, action.dest)}")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---------------------------- Build Datasets ----------------------------
    val_transformer = Augmentation(is_train=False, image_size=args.image_size, transforms=args.data_augment)
    val_dataset = VOCDataset(is_train = False,
                             data_dir = args.data_root,
                             transform = val_transformer,
                             image_set = args.val_dataset,
                             voc_classes = args.class_names,
                             )
    
    train_transformer = Augmentation(is_train=True, image_size=args.image_size, transforms=args.data_augment)
    train_dataset = VOCDataset(is_train = False,
                               data_dir = args.data_root,
                               transform = train_transformer,
                               image_set = args.train_dataset,
                               voc_classes = args.class_names,
                               )
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_b_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    # ----------------------- Build Model ----------------------------------------
    model = build_yolo(args, device, True)
    flops(model, args.image_size, device)
          
    criterion =  Loss(device = device,
                         anchor_size = args.anchor_size,
                         num_classes = args.num_classes,
                         boxes_per_cell = args.boxes_per_cell,
                         bbox_loss_weight = args.bbox_loss_weight,
                         objectness_loss_weight = args.objectness_loss_weight,
                         class_loss_weight = args.class_loss_weight)
    
    evaluator = Evaluator(
        device   =device,
        dataset  = val_dataset,
        ovthresh = args.nms_threshold,                        
        class_names = args.class_names,
        recall_thre = args.recall_threshold,
        visualization = args.eval_visualization)
    
    optimizer, start_epoch = build_optimizer(args, model)

    lr_scheduler, lf = build_lambda_lr_scheduler(args, optimizer)
    if args.resume_weight_path and args.resume_weight_path != 'None':
        lr_scheduler.last_epoch = start_epoch - 1  # do not move
        lr_scheduler.step()
    
    # ----------------------- Build Train ----------------------------------------
    max_mAP = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs_total):
        model.train()
        train_loss = 0.0
        model.trainable = True
        for iteration, (images, targets) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * len(train_dataloader)
            if epoch < args.warmup_epochs:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration,
                                           [0, args.warmup_epochs*len(train_dataloader)],
                                           [0.1 if j ==0 else 0.0, x['initial_lr'] * lf(epoch)])
                   
            ## forward
            images = images.to(device)
            outputs = model(images)

            ## loss
            loss_dict = criterion(outputs=outputs, targets=targets)
            [loss_obj, loss_cls, loss_box, losses] = loss_dict.values()
            if args.grad_accumulate > 1:
               losses /= args.grad_accumulate
               loss_obj /= args.grad_accumulate
               loss_cls /= args.grad_accumulate
               loss_box /= args.grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f}, Loss_obj: {:8.4f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epochs_total, iteration+1, len(train_dataloader), optimizer.param_groups[0]['lr'], losses, loss_obj, loss_cls, loss_box))
            train_loss += losses.item() * images.size(0)
        
        lr_scheduler.step()

        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        # save_model
        if epoch >= args.save_checkpoint_epoch:
            model.trainable = False
            model.nms_thresh = args.nms_threshold
            model.conf_thresh = args.confidence_threshold

            weight = '{}.pth'.format(epoch)
            ckpt_path = os.path.join(os.getcwd(), 'log', weight)
            if not os.path.exists(os.path.dirname(ckpt_path)): 
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                mAP = evaluator.eval(model)
            writer.add_scalar('mAP', mAP, epoch)
            print("Epoch [{}]".format('-'*100))
            print("Epoch [{}:{}], mAP [{:.4f}]".format(epoch, args.epochs_total, mAP))
            print("Epoch [{}]".format('-'*100))
            if mAP > max_mAP:
                torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'mAP':mAP,
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_mAP = mAP
        
if __name__ == "__main__":
    train()