# @Time 2025/1/26
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch.cuda
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import set_seed, create_logger, save_checkpoint, load_checkpoint, data_write_csv
from dataloader import get_dataset, mv_dataset, mv_tabular_collate
from Models.SeMo import SMiLE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--data_path", type=str, default="C:/Users/xiaol/Documents/BackupData/Code_data/Data/")
    parser.add_argument("--data_name", type=str, default="pancancer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--model", type=str, default="SeMo", choices=["baseline", "SeMo","bow", "img", "bert", "concatbow", "concatbert", "mmbt","latefusion"])
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--name", type=str, default="pancancer_SeMo_best")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--df", type=bool, default=False)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--conf_thr", type=float, default=0.95, help="conf_thr for predictions")
    parser.add_argument("--save_predicts", type=bool, default=False)
    parser.add_argument("--recover", action='store_true', help='whether load checkpoint')

def model_eval(data, model):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            xs, tgt, _ = batch

            for v in range(len(xs)):
                xs[v] = xs[v].float().to(device)
            tgt = tgt.long().to(device)

            out, outs = model(xs)
            pred = F.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {}
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    return metrics

def model_test(data, model, view, conf_thr):
    with torch.no_grad():
        N_modal_test = len(data) * view
        N_modal_use = 0
        preds, tgts = [], []
        for batch in data:
            xs, tgt, _ = batch
            for v in range(len(xs)):
                xs[v] = xs[v].float().to(device)
            tgt = tgt.long().to(device)
            use_modal, out = model.forward_test(xs, conf_thr)
            # print(use_modal)
            N_modal_use += use_modal
            pred = F.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
        # print("N_modal_use:", N_modal_use)
        use_rate = N_modal_use / N_modal_test
    metrics = {}
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["conf_thr"] = round(args.conf_thr, 5)  # float
    metrics["use_rate"] = round(use_rate, 5)
    metrics["acc"] = round(accuracy_score(tgts, preds), 5)
    return metrics, preds

def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    X_train, Y_train, X_test, Y_test, dims, n_view, class_num = \
        get_dataset(args.data_path, args.data_name)

    train_loader = DataLoader(
        mv_dataset(X_train, Y_train),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=mv_tabular_collate)

    test_loader = DataLoader(
        mv_dataset(X_test,Y_test),
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=mv_tabular_collate)

    model = SMiLE(n_view, dims, args.feature_dim, class_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, factor=args.lr_factor
    )
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
    if args.recover == True:
        if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
            start_epoch = checkpoint["epoch"]
            n_no_improve = checkpoint["n_no_improve"]
            best_metric = checkpoint["best_metric"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        preds, tgts = [], []
        for batch in tqdm(train_loader, total=len(train_loader)):
            data, tgt, _ = batch
            for v in range(len(data)):
                data[v] = data[v].float().to(device)
            tgt = tgt.long().to(device)
            out, outs = model(data)
            preds.append(out)
            tgts.append(tgt.cpu().detach().numpy())
            loss = criterion(out, tgt.long())
            for v in range(n_view):
                loss = loss + criterion(outs[v], tgt.long())
            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            optimizer.step()

        model.eval()
        metrics = model_eval(test_loader, model)
        print("val acc:", metrics["acc"])
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        tuning_metric = metrics["acc"]

        # # 打印调试信息
        # current_lr = optimizer.param_groups[0]["lr"]
        # print(
        #     f"Epoch {i_epoch + 1}: Tuning Metric = {tuning_metric:.4f}, "
        #     f"Best Metric = {best_metric:.4f}, LR = {current_lr:.6f}, "
        #     f"No Improvement Count = {n_no_improve}"
        # )

        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    # NOTE test-decision
    if args.recover == True:
        load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))

    # Introduce New modality
    conf_thr_start = 0.9999 # 初始化thr起始值
    conf_thr_step = 0.000005  # thr的步长
    conf_thr = np.arange(conf_thr_start, 1.01, conf_thr_step)
    # conf_thr = [1.01]
    filepath = args.savedir + f'_{n_view}_modalities.csv'
    for thr in conf_thr:
        args.conf_thr = thr
        test_metrics, preds = model_test(test_loader, model, n_view, args.conf_thr)
        data_write_csv(filepath, 'conf_thr: ' + str(test_metrics["conf_thr"]) + ' ' +
                   'acc: ' + str(test_metrics["acc"]) + ' '
                   + 'use_rate: ' + str(test_metrics["use_rate"]))
    data_write_csv(filepath, '*************')


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)