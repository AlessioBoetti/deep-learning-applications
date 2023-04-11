import sys
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from model import ConvolutionalNeuralNetworks
from utils import (get_loaders, save_checkpoint, load_checkpoint, 
                   load_best_model, metrics, eval_fn, 
                   create_directory_if_does_not_exist, EarlyStopping, Lion)


def train_fn(epoch, loader, model, optimizer, scheduler, loss_fn, scaler, metric_collection, device):
    model.train()
    running_loss = 0

    for (data, target) in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, target)

        # backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.updata()
        scheduler.step()
        running_loss += loss.item()
        metric_collection(prediction, target)

    train_loss = running_loss / len(loader)
    train_accuracy = metric_collection("MulticlassAccuracy").compute().cpu() * 100
    
    metric_collection.reset()

    return train_loss, train_accuracy


def main(wb, checkpoint_dir, weight_dir, device, num_workers):
    model = ConvolutionalNeuralNetworks(depth=wb.config['depth'],
                                        n_classes=wb.config['n_class'],
                                        want_shortcut=wb.config['want_shortcut'],
                                        pool_type=wb.config['pool_type']).to(device)
    
    optimizer = Lion(model.parameters(), l=wb.config['learning_rate'], weight_decay=wb.config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    metric_collection = metrics(wb, device)

    if wb.config['evaluation']:
        load_best_model(torch.load(weight_dir), model)
        test_loader = get_loaders(wb.config['batch_size'], num_workers, training=False)
        eval_fn(test_loader, model, criterion, metric_collection, device)
        sys.exit()

    if wb.resumed:
        start, monitored_value, count = load_checkpoint(torch.load(checkpoint_dir), model, optimizer)
        patience = EarlyStopping('max', wb.config['patience'], count, monitored_value)
    else:
        start = 0
        patience = EarlyStopping('max', wb.config['patience'])

    train_loader, test_loader = get_loaders(wb.config['batch_size'], num_workers)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=wb.config['max_lr'],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=wb.config['num_epochs'] - start)

    wb.watch(model, log="all")

    for epoch in range(start, wb.config['num_epochs']):
        train_loss, train_accuracy = train_fn(epoch, train_loader, model, optimizer, scheduler, criterion, scaler,
                                              metric_collection, device)
        
        test_loss, test_accuracy = eval_fn(test_loader, model, criterion, metric_collection, device)

        wb.log({'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
                })
        
        # save best model
        if patience(test_accuracy):
            wb.log({
                "accuracy_epoch": epoch,
            })
            checkpoint = {"state_dict": model.state_dict()}
            save_checkpoint("=> Best model found", checkpoint, weight_dir)

        # save checkpoint
        checkpoint = {
            'start': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'max_accuracy': getattr(patience, 'baseline'),
            'count': getattr(patience, 'count'),
        }
        save_checkpoint('=> Saving checkpoint', checkpoint, checkpoint_dir)

        # early stopping
        if getattr(patience, 'count') == 0:
            print('=> Patience finished')
            break

    sys.exit()


if __name__ == "__main__":
    wab = wandb.init(
        # set the wandb project where this run will be logged
        project="Rinse and Repeat",
        # group="Experiment",
        tags=[],
        resume=False,
        name="depth-48-skip",
        config={
            # model parameters
            "architecture": "Convolutional Neural Networks",
            "depth": 48,
            "n_class": 10,
            "want_shortcut": True,
            "pool_type": "max",

            # datasets
            "dataset": "CIFAR-10",

            # hyperparameters
            "learning_rate": 5e-5,
            "batch_size": 2048,
            "optimizer": "Lion",
            "weight_decay": 1e-2,
            "scheduler": "One Cycle Learning",
            "max_lr": 5e-4,
            "num_epochs": 200,
            "patience": 20,

            # run type
            "evaluation": False,
        })
    # local parameters
    check_dir = "".join(["checkpoint/", wab.name, "/"])
    w_dir = "".join(["result/", wab.name, "/"])
    create_directory_if_does_not_exist(check_dir, w_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers = 16
    main(wab, check_dir, w_dir, dev, n_workers)






    
