#-*- coding : utf-8-*-
import time
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import torch
import torch.nn as nn
from dataset_train import get_loader
from DualTalk import DualTalkModel

def trainer(args, train_loader, dev_loader, model, optimizer, criterion,epoch,last_train):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    log_file = os.path.join(save_path, 'training_log.txt')
    iteration = 0
    lr_list = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.8) #TODO
    for e in range(epoch):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        accumulation_steps = max(int(args.gradient_accumulation_steps), 1)
        optimizer.zero_grad(set_to_none=True)

        for i, (file_name, audio1, audio2, exp1, jawpose1, neck1, exp2, jawpose2, neck2) in pbar:
            iteration += 1
            audio1 = audio1.to(args.device)
            audio2 = audio2.to(args.device)
            exp1 = exp1.float().to(args.device) #[bs,frames,52]
            jawpose1 = jawpose1.float().to(args.device)
            neck1 = neck1.float().to(args.device)
            exp2 = exp2.float().to(args.device)
            jawpose2 = jawpose2.float().to(args.device)
            neck2 = neck2.float().to(args.device)
            if exp1.shape[1] > args.max_seq_len:
                exp1 = exp1[:,:args.max_seq_len,:]
                jawpose1 = jawpose1[:,:args.max_seq_len,:]
                neck1 = neck1[:,:args.max_seq_len,:]
                audio1 = audio1[:,:int(args.max_seq_len/25*16000)]
            if exp2.shape[1] > args.max_seq_len:
                exp2 = exp2[:,:args.max_seq_len,:]
                jawpose2 = jawpose2[:,:args.max_seq_len,:]
                neck2 = neck2[:,:args.max_seq_len,:]
                audio2 = audio2[:,:int(args.max_seq_len/25*16000)]
            blendshape1 = torch.cat((exp1, jawpose1, neck1), dim=2)
            blendshape2 = torch.cat((exp2, jawpose2, neck2), dim=2)
            bs_output = model(audio1,audio2,blendshape2)
            length = min(bs_output.shape[1],blendshape1.shape[1])
            bs_output = bs_output[:,:length,:]
            blendshape1 = blendshape1[:,:length,:]
            loss1 = criterion(bs_output[:,:,:50], blendshape1[:,:,:50])
            loss_jaw = criterion(bs_output[:,:, 50:53], blendshape1[:,:, 50:53])
            loss_neck = criterion(bs_output[:,:, 53:56], blendshape1[:,:, 53:56])
            gt_vel = blendshape1[:,1:, :] - blendshape1[:,:-1, :]
            pred_vel = bs_output[:,1:, :] - bs_output[:,:-1, :]
            loss3 = criterion(pred_vel, gt_vel)
            loss = torch.mean(loss1+loss_jaw+loss_neck+loss3)
            loss_log.append(loss.item())

            # Average the gradients over one accumulation group. The last
            # group may contain fewer micro-batches than accumulation_steps.
            group_start = (i // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            (loss / group_size).backward()

            should_step = (
                (i + 1) % accumulation_steps == 0
                or (i + 1) == len(train_loader)
            )
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f} EXP LOSS:{:.4f} JAW LOSS:{:.4f} NECK LOSS:{:.4f} VELOCITY LOSS:{:.4f} LR:{:.7f}".format((e+1+last_train), iteration ,np.mean(loss_log),loss1.item(),loss_jaw.item(),loss_neck.item(),loss3.item(),scheduler.get_last_lr()[0]))

        val_loss_log = []
        model.eval()
        with torch.no_grad():
            for file_name, audio1, audio2, exp1, jawpose1, neck1, exp2, jawpose2, neck2 in dev_loader:
                audio1 = audio1.to(args.device)
                audio2 = audio2.to(args.device)
                exp1 = exp1.float().to(args.device) 
                jawpose1 = jawpose1.float().to(args.device)
                neck1 = neck1.float().to(args.device)
                exp2 = exp2.float().to(args.device)
                jawpose2 = jawpose2.float().to(args.device)
                neck2 = neck2.float().to(args.device)
                if exp1.shape[1] > args.max_seq_len:
                    exp1 = exp1[:,:args.max_seq_len,:]
                    jawpose1 = jawpose1[:,:args.max_seq_len,:]
                    neck1 = neck1[:,:args.max_seq_len,:]
                    audio1 = audio1[:,:int(args.max_seq_len/25*16000)]
                if exp2.shape[1] > args.max_seq_len:
                    exp2 = exp2[:,:args.max_seq_len,:]
                    jawpose2 = jawpose2[:,:args.max_seq_len,:]
                    neck2 = neck2[:,:args.max_seq_len,:]
                    audio2 = audio2[:,:int(args.max_seq_len/25*16000)]
                blendshape1 = torch.cat((exp1, jawpose1, neck1), dim=2)
                blendshape2 = torch.cat((exp2, jawpose2, neck2), dim=2)
                bs_output = model(audio1,audio2,blendshape2)
                length = min(bs_output.shape[1],blendshape1.shape[1])
                bs_output = bs_output[:,:length,:]
                blendshape1 = blendshape1[:,:length,:]
                val_exp = criterion(bs_output[:,:,:50], blendshape1[:,:,:50])
                val_jaw = criterion(bs_output[:,:, 50:53], blendshape1[:,:, 50:53])
                val_neck = criterion(bs_output[:,:, 53:56], blendshape1[:,:, 53:56])
                val_loss_log.append(val_exp.item()+val_jaw.item()+val_neck.item())

        print("epoch {} val all loss:{:.7f}, exp loss: {:.7f}, jaw loss: {:.7f}, neck loss: {:.7f}".format(e+1, np.mean(val_loss_log), val_exp.item(), val_jaw.item(), val_neck.item()))

        scheduler.step()
        lr_list.append(scheduler.get_last_lr())
        if (e%10 ==0 and e>=1):
            torch.save(model.state_dict(), os.path.join(save_path, "model_{}.pth".format(e+last_train)))
            print("Model saved at epoch {}".format(e+last_train))
        
        with open(log_file, 'a') as f:
            f.write(f"Epoch {e+1+last_train}, "
                    f"Train Loss: {np.mean(loss_log):.7f}, "
                    f"Train Exp Loss: {loss1.item():.7f}, "
                    f"Train Jaw Loss: {loss_jaw.item():.7f}, "
                    f"Train Neck Loss: {loss_neck.item():.7f}, "
                    f"Val Loss: {np.mean(val_loss_log):.7f}, "
                    f"Val Exp Loss: {val_exp.item():.7f}, "
                    f"Val Jaw Loss: {val_jaw.item():.7f}, "
                    f"Val Neck Loss: {val_neck.item():.7f}, "
                    f"Learning Rate: {scheduler.get_last_lr()[0]:.7f}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--train_data_path", type=str, default= "./data/train/", help='path of the training data')
    parser.add_argument("--val_data_path", type=str, default= "./data/test/", help='path of the validation data')
    parser.add_argument("--test_data_path", type=str, default= None, help='path of the test data')
    parser.add_argument("--seed", type=int, default=6666, help='random seed')
    parser.add_argument("--scale",type=str, default="large",help="large or base")
    parser.add_argument("--blendshape_dim", type=int, default=56, help='number of blendshapes:52')
    parser.add_argument("--feature_dim", type=int, default=256, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--device", type=str, default="cuda:1", help='cuda:0 or cuda:1')
    parser.add_argument("--save_path", type=str, default="save_DualTalk", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result_DualTalk", help='path to the predictions')
    parser.add_argument("--load_path", type=str, default=None, help='path to the trained models')
    parser.add_argument("--max_seq_len", type=int, default=1800, help='max_seq_len')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    model = DualTalkModel(args)
    model = model.to(args.device)
    dataset = get_loader(args)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model = trainer(args, dataset['train'],dataset['val'],model, optimizer,criterion, args.max_epoch,args.last_train)

if __name__ == "__main__":
    main()
