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

gt_train = "./transformer/data/gt_split/train.tsv"
gt_validation = "./transformer/data/gt_split/validation.tsv"
tokensfile = "./transformer/data/tokens.tsv"
root = "./transformer/data/train/"
checkpoint_path = "./checkpoints"

imgWidth = 256
imgHeight = 256

SoftMax = nn.Softmax(dim=2)

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
        # x = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
        # y = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

        x = torch.tensor(batch["image"], device=device)
        y = torch.tensor(batch["truth"]["encoded"], device=device)
        y[y == -1] = trg_pad_idx

        # Shift tgt input by 1 for prediction
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Training with y_input and tg_mask
        pred = model(x, y_input)

        # Softmax
        pred = SoftMax(pred) # TODO: remove softmax layer and add it to the model
        pred = pred.permute((0,2,1))
        loss = loss_fn(pred, y_expected)

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
            # x = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            # y = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

            x = torch.tensor(batch["image"], device=device)
            y = torch.tensor(batch["truth"]["encoded"], device=device)
            y[y == -1] = trg_pad_idx


            # Shift tgt input by 1 for prediction
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Training with y_input
            pred = model(x, y_input)
            pred = SoftMax(pred) # TODO: remove softmax layer and add it to the model
            pred = pred.permute((0,2,1))
            
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    start_epoch = 0

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    model = nn.DataParallel(model) # Comment out if using only one GPU
    
    for epoch in range(start_epoch, epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        start = datetime.now()
        
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
            #'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, checkpoint_path)

        end = datetime.now()
        print("Elapsed Time: ", (end-start).total_seconds(), "s")
    
    return train_loss_list, validation_loss_list


def predict(model, device, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        #tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()
  
  
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

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    #train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_data_loader, validation_data_loader, epochs=10, device=device)
    fit(model, opt, loss_fn, train_data_loader, validation_data_loader, epochs=10, device=device)

if __name__ == "__main__":
    """
    example code to verify functionality of Transformer
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

