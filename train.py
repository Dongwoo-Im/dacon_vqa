import time
import wandb
import argparse
import pandas as pd
import os.path as osp
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.trainer_utils import seed_worker
from transformers import BlipConfig, BlipProcessor, BlipImageProcessor, BertTokenizerFast, BlipForQuestionAnswering

from dataset import VQADataset
import utils
try:
    import wandb_utils
except:
    wandb_utils = None

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--work_dir', type=str, default='./work_dirs')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    # Log
    parser.add_argument('--use_wandb', type=utils.str2bool, default=False)
    # Data
    parser.add_argument('--df_ver', type=int, default=1)
    parser.add_argument('--fold', type=int, default=-1)
    # Train
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--grad_accum', type=int, default=4)
    # Model
    parser.add_argument('--pretrained_ckpt', type=str, default='model_base.pth', choices=['model_base.pth', 'model_base_capfilt_large.pth'])
    parser.add_argument('--freeze_image_encoder', type=utils.str2bool, default=True)
    args = parser.parse_args()
    return args


def train(epoch, model, loader, optimizer, scheduler, scaler, args, log_freq=1000):
    start = end = time.time()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.train()
    optimizer.zero_grad()
    
    if args.freeze_image_encoder:
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = False

    len_loader = len(loader)
    for i, inputs in enumerate(loader):
        data_time.update(time.time() - end)
        
        with autocast():
            for k in inputs.keys():
                inputs[k] = inputs[k].to(args.device)
            
            outputs = model(**inputs)
            
            loss = outputs.loss / args.grad_accum
            losses.update(loss.item())

        scaler.scale(loss).backward()
        if (i+1) % args.grad_accum == 0 or (i+1) == len_loader:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0 * args.grad_accum)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % log_freq == 0 or (i+1) == len_loader:
            print(
                'Epoch {0} [{1}/{2}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Elapsed {remain:s} '
                'Loss: {loss_val:.3f}({loss_avg:.3f}) '
                .format(
                    epoch, i+1, len_loader,
                    data_time=data_time,
                    remain=utils.timeSince(start, float(i+1)/len_loader),
                    loss_val=losses.val * args.grad_accum,
                    loss_avg=losses.avg * args.grad_accum,
                )
            )
        
        if args.use_wandb:
            wandb.log({
                'train_loss': round(losses.val * args.grad_accum, 4),
                'learning_rate': scheduler.optimizer.param_groups[0]['lr'],
            })

    return round(losses.avg * args.grad_accum, 4)


@torch.no_grad()
def evaluation(model, loader, processor, args, device):
    total_bs = 0
    total_correct = 0
    total_correct_new = 0
    
    model.eval()

    pbar = tqdm(loader, total=len(loader))
    for inputs in pbar:

        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model.generate(**inputs)
        pred = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        gt = processor.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
        bs = inputs['labels'].size(0)
        
        total_bs += bs
        total_correct += sum([g==p for g,p in zip(gt, pred)])

        pbar.set_postfix(
            acc=total_correct/total_bs, 
            acc_new=total_correct_new/total_bs,
        )
    
    acc = round(total_correct/total_bs, 4)
    if args.use_wandb:
        wandb.log({'valid_accuracy': acc})
    return acc


def main(args):
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # DataFrame
    df = pd.read_csv(f'data/train_5fold_ver{args.df_ver}.csv')
    if args.fold == -1:
        train_df = df
    else:
        train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
        valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    # Model
    image_processor = BlipImageProcessor()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    processor = BlipProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model_config = BlipConfig()
    model = BlipForQuestionAnswering(model_config).to(args.device)
    
    # pretrained weight from (129M & BLIP w/ ViT-B)
    # https://github.com/salesforce/BLIP#pre-trained-checkpoints
    state_dict = torch.load(args.pretrained_ckpt)['model']
    for key in state_dict.copy():
        value = state_dict.pop(key)
        if key == 'visual_encoder.pos_embed':
            value = utils.interpolate_pos_embed(value, model.vision_model)
        renamed_key = utils.rename_key(key)
        state_dict[renamed_key] = value
    model.load_state_dict(state_dict, strict=False)
    
    model.gradient_checkpointing_enable()

    # Dataset & Dataloader
    loader_dict = {"pin_memory": True, "num_workers": 4, "worker_init_fn": seed_worker}
    train_set = VQADataset(train_df, processor, mode='train')
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        **loader_dict
    )
    if args.fold != -1:
        if args.freeze_image_encoder:
            valid_batch_size = 32
        else:
            valid_batch_size = 16
        valid_set = VQADataset(valid_df, processor, mode='valid')
        valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, **loader_dict)

    # Optimizer & Scheduler & GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.num_epochs / args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    scaler = GradScaler()

    # Training loop
    valid_acc = 0
    for epoch in range(1, args.num_epochs + 1):
        print('-' * 10)
        print(f'Epoch {epoch} / {args.num_epochs}')

        train_loss = train(epoch, model, train_loader, optimizer, scheduler, scaler, args)
        
        if args.fold != -1:
            valid_acc = evaluation(model, valid_loader, processor, args, args.device)
        
        if epoch % args.save_freq == 0:
            file_name = f'epoch{epoch}_acc{valid_acc}.pt'
            torch.save(model.state_dict(), osp.join(args.work_dir_exp, file_name))
        
        print(f'[Epoch {epoch}] [Train] Loss:{train_loss}')
        if args.fold != -1:
            print(f'[Epoch {epoch}] [Valid] Acc:{valid_acc}')


if __name__ == "__main__":
    args = get_parser()
    args.work_dir_exp = utils.get_exp_dir(args.work_dir)
    args.config_dir = osp.join(args.work_dir_exp, 'config.yaml')
    utils.save_config(args, args.config_dir)
    utils.set_seeds(args.seed)
    if args.use_wandb:
        wandb_utils.wandb_init(args)
    main(args)
