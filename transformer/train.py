import torch
import torch.nn as nn
import numpy as np
import random
from dataset import CrohmeDataset, START, PAD, collate_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing
from transformer_vtex import Transformer

gt_train = "./transformer/data/gt_split/train.tsv"
gt_validation = "./transformer/data/gt_split/validation.tsv"
tokensfile = "./transformer/data/tokens.tsv"
root = "./transformer/data/train/"

trg_vocab_size = 101
trg_pad_idx = 0 # What index in the dictory is the pad character, 2 is EOS character
max_trg_length = 100
img_height = 256
img_width = 256

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize((img_width, img_height)),
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

        x = torch.tensor(batch["image"], dtype=torch.long, device=device)
        y = torch.tensor(batch["truth"]["encoded"], dtype=torch.long, device=device)

        # Shift tgt input by 1 for prediction
        y_input = y[:, :-1]
        y_expected = y[:, 1:]


        # Training with y_input and tg_mask
        pred = model(x, y_input)

        # Premute pred to have batch size first
        pred = pred.permute(1,2,0)
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

            x = torch.tensor(batch["image"], dtype=torch.long, device=device)
            y = torch.tensor(batch["truth"]["encoded"], dtype=torch.long, device=device)

            # Shift tgt input by 1 for prediction
            y_input = y[:, :-1]
            y_expected = y[:, 1:]


            # Training with y_input
            pred = model(x, y_input)

            # Premute pred to have batch size first
            pred = pred.permute(1,2,0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
    
    return train_loss_list, validation_loss_list


def predict(model, device, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
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
        batch_size=4,
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
        batch_size=4,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=collate_batch,
    )

    print("Loaded Validation Dataset")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # train_data = train.generate_random_data(9000)
    # val_data = train.generate_random_data(3000)

    # train_dataloader = train.batchify_data(train_data)
    # val_dataloader = train.batchify_data(val_data)

    #train_loss_list, validation_loss_list = train.fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10, device)
    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_data_loader, validation_data_loader, epochs=10, device=device)





if __name__ == "__main__":
    """
    example code to verify functionality of Transformer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # new src [2, 1, 1000, 1000]
    src1 = torch.rand(1000, 1000).unsqueeze(0).to(device)
    src2 = torch.rand(1000, 1000).unsqueeze(0).to(device)
    src = torch.stack((src1, src2), dim=0)

    trg = torch.tensor([[1,7,3,4,7,2,0],[1,4,3,5,7,9,2]]).to(device)


    model = Transformer(device, trg_vocab_size, trg_pad_idx, max_trg_length, img_height, img_width).to(device)

    #out = model(src, trg[:, :-1])
    #print(out.shape)

    train(model, device)





