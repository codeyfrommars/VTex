import torch
import torch.nn as nn
from dataset import CrohmeDataset, PAD, collate_batch, gt_train, gt_validation, tokensfile, root_train, train_checkpoint_path, transformers
from torch.utils.data import DataLoader
import multiprocessing
import transformer_vtex
from datetime import datetime
import matplotlib.pyplot as plt

# Unzip data.zip and put the correct paths to the training data

def train_loop(model, opt, loss_fn, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x = torch.tensor(batch["image"], device=device)
        y = torch.tensor(batch["truth"]["encoded"], device=device)

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

    start_epoch = 0

    # Load checkpoint
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(), # model.state_dict() if using only one GPU
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            }, train_checkpoint_path)

        end = datetime.now()
        print("Elapsed Time: ", (end-start).total_seconds(), "s")
        
        plt.figure(figsize=(15, 15))
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_list, train_loss_list, color ="green", label="Training Loss", marker='o', markerfacecolor='green')
        plt.plot(epoch_list, validation_loss_list, color ="red", linewidth=1.0, linestyle='--', label="Validation Loss", marker='o', markerfacecolor='red')

        plt.xticks(ticks=epoch_list)
        plt.legend()
        plt.savefig('./progress/progress_epoch_'+str(epoch+1)+'.png')
    
    return train_loss_list, validation_loss_list
  
  
def train(model, device):
    train_dataset = CrohmeDataset(
        gt_train, tokensfile, root=root_train, crop=False, transform=transformers
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=6,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=collate_batch,
    )
    print("Loaded Train Dataset")

    validation_dataset = CrohmeDataset(
        gt_validation, tokensfile, root=root_train, crop=False, transform=transformers
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=6,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=collate_batch,
    )

    print("Loaded Validation Dataset")

    opt = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-04)
    loss_fn = nn.CrossEntropyLoss()
    fit(model, opt, loss_fn, train_data_loader, validation_data_loader, epochs=5000, device=device)

if __name__ == "__main__":
    """
    code to train transformer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Crohme Dataset to get the <EOS>
    train_dataset = CrohmeDataset(gt_train, tokensfile, root=root_train, crop=False)
    trg_pad_idx = train_dataset.token_to_id[PAD]
    trg_vocab_size =  len(train_dataset.token_to_id)
    max_trg_length = 55
    max_img_size = 500

    # Initialize model
    model = transformer_vtex.Transformer(device, trg_vocab_size, trg_pad_idx, max_trg_length).to(device)
    train(model, device)

