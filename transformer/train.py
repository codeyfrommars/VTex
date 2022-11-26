import torch
import torch.nn as nn
import numpy as np
import random
from dataset import CrohmeDataset, START, PAD, collate_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing

gt_train = "./transformer/data/gt_split/train.tsv"
gt_validation = "./transformer/data/gt_split/validation.tsv"
tokensfile = "./transformer/data/tokens.tsv"
root = "./transformer/data/train/"

imgWidth = 500
imgHeight = 500

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

        x = torch.tensor(batch["Image"], dtype=torch.long, device=device)
        y = torch.tensor(batch["Truth"], dtype=torch.long, device=device)

        # Shift tgt input by 1 for prediction
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask tgt input
        sequence_length = y_input.size(1)
        # TODO: y_input or size?
        tgt_mask = model._make_trg_mask(y_input).to(device)

        # Training with y_input and tg_mask
        pred = model(x, y_input, tgt_mask)

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

            x = torch.tensor(batch["Image"], dtype=torch.long, device=device)
            y = torch.tensor(batch["Truth"], dtype=torch.long, device=device)

            # Shift tgt input by 1 for prediction
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask tgt input
            sequence_length = y_input.size(1)
            # TODO: y_input or size?
            tgt_mask = model._make_trg_mask(y_input).to(device)

            # Training with y_input and tg_mask
            pred = model(x, y_input, tgt_mask)

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

    # # Here we test some examples to observe how the model predicts
    # examples = [
    #     torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
    #     torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
    #     torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    #     torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
    #     torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    #     torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
    # ]

    # for idx, example in enumerate(examples):
    #     result = train.predict(model, device, example)
    #     print(f"Example {idx}")
    #     print(f"Input: {example.view(-1).tolist()[1:-1]}")
    #     print(f"Continuation: {result[1:-1]}")
    #     print()



    
# def generate_random_data(n):
#     SOS_token = np.array([2])
#     EOS_token = np.array([3])
#     length = 8

#     data = []

#     # 1,1,1,1,1,1 -> 1,1,1,1,1
#     for i in range(n // 3):
#         X = np.concatenate((SOS_token, np.ones(length), EOS_token))
#         y = np.concatenate((SOS_token, np.ones(length), EOS_token))
#         data.append([X, y])

#     # 0,0,0,0 -> 0,0,0,0
#     for i in range(n // 3):
#         X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
#         y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
#         data.append([X, y])

#     # 1,0,1,0 -> 1,0,1,0,1
#     for i in range(n // 3):
#         X = np.zeros(length)
#         start = random.randint(0, 1)

#         X[start::2] = 1

#         y = np.zeros(length)
#         if X[-1] == 0:
#             y[::2] = 1
#         else:
#             y[1::2] = 1

#         X = np.concatenate((SOS_token, X, EOS_token))
#         y = np.concatenate((SOS_token, y, EOS_token))

#         data.append([X, y])

#     np.random.shuffle(data)

#     return data


# def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
#     batches = []
#     for idx in range(0, len(data), batch_size):
#         # We make sure we dont get the last bit if its not batch_size size
#         if idx + batch_size < len(data):
#             # Here you would need to get the max length of the batch,
#             # and normalize the length with the PAD token.
#             if padding:
#                 max_batch_length = 0

#                 # Get longest sentence in batch
#                 for seq in data[idx : idx + batch_size]:
#                     if len(seq) > max_batch_length:
#                         max_batch_length = len(seq)

#                 # Append X padding tokens until it reaches the max length
#                 for seq_idx in range(batch_size):
#                     remaining_length = max_batch_length - len(data[idx + seq_idx])
#                     data[idx + seq_idx] += [padding_token] * remaining_length

#             batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

#     print(f"{len(batches)} batches of size {batch_size}")

#     return batches

