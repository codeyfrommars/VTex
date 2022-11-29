import torch
import torch.nn as nn
import numpy as np
import random
from dataset import CrohmeDataset, START, PAD, collate_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing
import transformer_vtex
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

gt_train = "./transformer/data/gt_split/train.tsv"
gt_validation = "./transformer/data/gt_split/validation.tsv"
tokensfile = "./transformer/data/tokens.tsv"
root = "./transformer/data/train/"
checkpoint_path = "./checkpoints"

imgWidth = 256
imgHeight = 256

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize((imgWidth, imgHeight)),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize([0.5], [0.5])
    ]
)

def train_loop(model, opt, loss_fn, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x = torch.tensor(batch["image"], device=device)
        y = torch.tensor(batch["truth"]["encoded"], device=device)
        y[y == -1] = trg_pad_idx

        # Shift tgt input by 1 for prediction
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Training with y_input and tg_mask
        pred = model(x, y_input) # [batch_size, seq_len, classes]
        
        # flatten tensors for cross-entropy loss
        pred_flat = pred.contiguous().view(-1, pred.shape[-1]) # [(batch_size seq_len), classes]
        y_expected_flat = y_expected.contiguous().view(-1) # [(batch_size seq_len)]
        loss = loss_fn(pred_flat, y_expected_flat)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x = torch.tensor(batch["image"], device=device)
            y = torch.tensor(batch["truth"]["encoded"], device=device)
            y[y == -1] = trg_pad_idx


            # Shift tgt input by 1 for prediction
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Training with y_input
            pred = model(x, y_input)

            # flatten tensors for cross-entropy loss
            pred_flat = pred.contiguous().view(-1, pred.shape[-1]) # [(batch_size seq_len), classes]
            y_expected_flat = y_expected.contiguous().view(-1) # [(batch_size seq_len)]
            loss = loss_fn(pred_flat, y_expected_flat)
            
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device):
    # Used for plotting later on
    train_loss_list, validation_loss_list, epoch_list = [], [], []

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    start_epoch = 0

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    model = nn.DataParallel(model) # Comment out if using only one GPU
    
    for epoch in range(start_epoch, epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        start = datetime.now()
        epoch_list += [epoch+1]

        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

        # Save checkpoint
        # filename = "{prefix}{num:0>4}.pth".format(num=checkpoint["epoch"], prefix="checkpoint")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(), # model.state_dict() if using only one GPU
            #'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            }, checkpoint_path)

        end = datetime.now()
        print("Elapsed Time: ", (end-start).total_seconds(), "s")
        
        #plt.figure(figsize=(15, 15))
        plt.plot(epoch_list[-20:], train_loss_list[-20:], color ="green", label="Training Loss", marker='o', markerfacecolor='green')
        plt.plot(epoch_list[-20:], validation_loss_list[-20:], color ="red", linewidth=1.0, linestyle='--', label="Validation Loss", marker='o', markerfacecolor='red')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.xticks(ticks=epoch_list[-20:])
        if epoch == start_epoch:
            plt.legend()
        #plt.legend()
        #plt.show()
        plt.savefig('./progress/progress_epoch_'+str(epoch+1)+'.png')
    
    return train_loss_list, validation_loss_list
  
  
def train(model, device):
    train_dataset = CrohmeDataset(
        gt_train, tokensfile, root=root, crop=False, transform=transformers
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=collate_batch,
    )
    print("Loaded Train Dataset")

    validation_dataset = CrohmeDataset(
        gt_validation, tokensfile, root=root, crop=False, transform=transformers
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=collate_batch,
    )

    print("Loaded Validation Dataset")

    opt = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-04)
    loss_fn = nn.CrossEntropyLoss()
    #train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_data_loader, validation_data_loader, epochs=10, device=device)
    fit(model, opt, loss_fn, train_data_loader, validation_data_loader, epochs=5000, device=device)

if __name__ == "__main__":
    """
    code to train transformer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Crohme Dataset to get the <EOS>
    train_dataset = CrohmeDataset(gt_train, tokensfile, root=root, crop=False)
    trg_pad_idx = train_dataset.token_to_id[PAD]
    trg_vocab_size =  len(train_dataset.token_to_id)
    max_trg_length = 100

    # Initialize model
    model = transformer_vtex.Transformer(device, trg_vocab_size, trg_pad_idx, max_trg_length, imgHeight, imgWidth).to(device)
    train(model, device)

