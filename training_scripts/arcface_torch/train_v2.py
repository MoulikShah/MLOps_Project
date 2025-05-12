import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import torch.nn.functional as F


assert torch.__version__ >= "1.12.0"

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

import json
import shutil

def create_symlink_dataset(full_data_root, sampled_json_path, output_symlink_root):
    with open(sampled_json_path) as f:
        sampled_classes = set(json.load(f)["sampled_classes"])

    os.makedirs(output_symlink_root, exist_ok=True)

    print(f"Linking {len(sampled_classes)} classes from {full_data_root} to {output_symlink_root}...")
    linked_count = 0
    for class_name in sampled_classes:
        src = os.path.join(full_data_root, class_name)
        dst = os.path.join(output_symlink_root, class_name)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            linked_count += 1

    print(f"✅ Linked {linked_count} class folders.")
    return output_symlink_root



def main(args):
    assert torch.cuda.is_available()
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard")) if rank == 0 else None

    # MLFlow Setup
    if rank == 0:
        try:
            mlflow.end_run()
        except:
            pass
        finally:
            mlflow.set_tracking_uri("http://129.114.27.48:8000")
            mlflow_experiment_name = datetime.now().strftime("%y%m%d_%H%M")
            mlflow_experiment_name = mlflow_experiment_name if cfg.experiment_name is None else mlflow_experiment_name + f"_{cfg.experiment_name}"
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow.start_run(log_system_metrics=True)
            mlflow.log_params(cfg)
            try:
                gpu_info = os.popen("nvidia-smi").read()
                mlflow.log_text(gpu_info, "gpu-info.txt")
            except:
                pass

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")

        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    
    # Create reduced dataset view
    reduced_data_path = "/home/cc/dataset/"  # or any temp path
    filtered_root = create_symlink_dataset(cfg.rec, cfg.class_json, reduced_data_path)

    train_loader = get_dataloader(filtered_root, local_rank, cfg.batch_size, cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers, cfg.class_json)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16, find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(64, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2], cfg.interclass_filtering_threshold)

    module_partial_fc = PartialFC_V2(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
    module_partial_fc.train().cuda()

    opt_cls = torch.optim.SGD if cfg.optimizer == "sgd" else torch.optim.AdamW
    opt = opt_cls(params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(optimizer=opt, warmup_iters=cfg.warmup_step, total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    callback_verification = CallBackVerification(val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer, wandb_logger=wandb_logger)
    callback_logging = CallBackLogging(frequent=cfg.frequent, total_step=cfg.total_step, batch_size=cfg.batch_size, start_step=global_step, writer=summary_writer)

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    start_time = time.time()
    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                if rank == 0:
                    mlflow.log_metrics({"step_loss": loss.item(), "avg_train_loss": loss_am.avg}, step=global_step)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)
            mlflow.pytorch.log_model(backbone.module, "epoch{}_model".format(epoch))

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            train_loader.reset()
    end_time = time.time()
    training_duration = end_time - start_time

    reduced_data_path_test = "/home/cc/dataset_test/"  # or any temp path
    filtered_root_test = create_symlink_dataset(cfg.test, cfg.class_json, reduced_data_path_test)

    test_loader = get_dataloader(
        filtered_root_test, local_rank=local_rank, batch_size=cfg.batch_size, dali=cfg.dali,
        dali_aug=False, seed=cfg.seed, num_workers=cfg.num_workers, sampled_classes_json=cfg.class_json
    )

    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                # outputs = model(images)
                embeddings = model(images)
                outputs = module_partial_fc.forward_test(embeddings)
                loss = F.cross_entropy(outputs, labels)
                loss_sum += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = loss_sum / total
        accuracy = correct / total
        return avg_loss, accuracy


    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        mlflow.pytorch.log_model(backbone.module, "final_model")
        test_loss, test_acc = evaluate_model(backbone.module, test_loader)
        mlflow.log_metrics({"training_time": training_duration})
        mlflow.log_metrics({
            "train_loss": test_loss,
            "train_accuracy": test_acc
        })
        mlflow.end_run()

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
