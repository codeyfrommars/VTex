import torch
import torch.nn as nn
import numpy as np
import random
from dataset import CrohmeDataset, START, PAD, END, collate_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing
import transformer_vtex
import matplotlib.pyplot as plt
import editdistance

# Update these to use different test dataset
gt_test = "./transformer/data/groundtruth_2016.tsv"
tokensfile = "./transformer/data/tokens.tsv"
root = "./transformer/data/test/2016/"
checkpoint_path = "./checkpoints"

imgWidth = 256
imgHeight = 256

max_trg_length = 100

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize((imgWidth, imgHeight)),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize([0.5], [0.5])
    ]
)

class ExpRate:
    def __init__(self):
        self.total = 0
        self.rec = 0
    
    def update(self, prediction, expected):
        dist = editdistance.eval(prediction, expected)
        if (dist == 0):
            self.rec += 1
        self.total += 1
    
    def compute(self) -> float:
        return self.rec / self.total

    

def greedy(model, device):
    """
    Use greedy method to predict sentence
    """
    test_dataset = CrohmeDataset(
        gt_test, tokensfile, root=root, crop=False, transform=transformers
    )

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded Checkpoint")

    model.eval()

    for item in test_dataset:
        src = torch.tensor(item["image"], device=device)
        src = src.unsqueeze(dim=1)
        truth = item["truth"]

        sos_idx = test_dataset.token_to_id[START]
        eos_idx = test_dataset.token_to_id[END]

        with torch.no_grad():
            # Predict the target sequence
            output = torch.tensor([[sos_idx]], dtype=torch.long, device=device) # [batch_size, len]
            for _ in range(max_trg_length):
                # predict the next token
                pred = model(src, output) # [batch_size, seq_len, classes]

                # perform argmax on the last 
                next_token = torch.argmax(pred, dim=-1) #[batch_size, len]
                next_token = next_token[:,-1].unsqueeze(dim=1) # [batch_size, 1]

                # append next token to output
                output = torch.cat([output, next_token], dim=1) # [batch_size, len+1]

                # stop if model predicts EOS
                if next_token.item()==eos_idx:
                    break
            
            # Remove SOS from sentence
            print ("Expected: " + str(truth))
            print ("Output: " + str(output))
            output_text = ""
            for i in output[0]:
                output_text = output_text + test_dataset.id_to_token[i.item()]

def beam_search(model: transformer_vtex.Transformer, device):
    """
    Use beam search to predict sentence (preferred)
    """
    test_dataset = CrohmeDataset(
        gt_test, tokensfile, root=root, crop=False, transform=transformers
    )
    sos_idx = test_dataset.token_to_id[START]
    eos_idx = test_dataset.token_to_id[END]
    pad_idx = test_dataset.token_to_id[PAD]

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded Checkpoint")

    model.eval()

    for item in test_dataset:
        src = torch.tensor(item["image"], device=device)
        src = src.unsqueeze(dim=1)
        truth = item["truth"]

        with torch.no_grad():
            output = model.beam_search(src, pad_idx, sos_idx, eos_idx, beam_size=10)

            print ("Expected: " + str(truth))
            print ("Output: " + str(output))
            output_text = ""
            for i in output:
                output_text = output_text + test_dataset.id_to_token[i.item()]
            print ("Output text: " + output_text)

def test(model, device, beam_size=10):
    """
    Perform beam search on test dataset and return the expression recognition rate
    """
    # load test dataset
    test_dataset = CrohmeDataset(
        gt_test, tokensfile, root=root, crop=False, transform=transformers
    )
    sos_idx = test_dataset.token_to_id[START]
    eos_idx = test_dataset.token_to_id[END]
    pad_idx = test_dataset.token_to_id[PAD]
    print("Loaded Test Dataset")

    # Create evaluation object
    exprate = ExpRate()

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded Checkpoint")

    model.eval()

    for item in test_dataset:
        src = torch.tensor(item["image"], device=device)
        src = src.unsqueeze(dim=1)
        truth = item["truth"]

        with torch.no_grad():
            output = model.beam_search(src, pad_idx, sos_idx, eos_idx, beam_size)

            exprate.update(output.tolist(), truth["encoded"])

            print ("Expected: " + str(truth))
            print ("Output: " + str(output))

    # Calculate score
    return exprate.compute()




if __name__ == "__main__":
    """
    code to test transformer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Crohme Dataset to get the <EOS>
    train_dataset = CrohmeDataset(gt_test, tokensfile, root=root, crop=False)
    trg_pad_idx = train_dataset.token_to_id[PAD]
    eos_idx = train_dataset.token_to_id[END]
    trg_vocab_size =  len(train_dataset.token_to_id)

    # Initialize model
    model = transformer_vtex.Transformer(device, trg_vocab_size, trg_pad_idx, max_trg_length, imgHeight, imgWidth).to(device)
    exprate = test(model, device)
    print("ExpRate: " + str(exprate * 100) + "%")