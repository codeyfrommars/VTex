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

# Unzip data.zip and put the correct paths to the test data
gt_test = "./transformer/data2/groundtruth_2019.txt"
tokensfile = "./transformer/data2/tokens.txt"
root = "./transformer/data2/2019/"
checkpoint_path = "./checkpoints_bttr_data500"


max_trg_length = 55

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        # transforms.Resize((imgWidth, imgHeight)),
        transforms.ToTensor(), # normalize to [0,1]
        # normalize
        # transforms.Normalize([0.5], [0.5])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

class ExpRate:
    def __init__(self):
        self.total = 0
        self.zero = 0
        self.one = 0
        self.five = 0
        self.ten = 0
    
    def update(self, prediction, expected):
        dist = editdistance.eval(prediction, expected)
        if (dist == 0):
            self.zero += 1
            self.one += 1
            self.five += 1
            self.ten += 1
        elif (dist <= 1):
            self.one += 1
            self.five += 1
            self.ten += 1
        elif (dist <= 5):
            self.five += 1
            self.ten += 1
        elif (dist <= 10):
            self.ten += 1
        self.total += 1
    
    def computeZero(self) -> float:
        return self.zero / self.total
    def computeOne(self) -> float:
        return self.one / self.total
    def computeFive(self) -> float:
        return self.five / self.total
    def computeTen(self) -> float:
        return self.ten / self.total

    

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

            print ("Expected: " + str(truth["text"]))
            # print ("Output: " + str(output))
            output_text = ""
            for i in output:
                output_text = output_text + test_dataset.id_to_token[i.item()]
            print ("Output:   " + output_text)

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

            print ("Expected: " + str(truth["text"]))
            # print ("Output: " + str(output))
            output_text = ""
            for i in output:
                if i.item() != sos_idx and i.item() != eos_idx:
                    output_text = output_text + test_dataset.id_to_token[i.item()]
            print ("Output:   " + output_text)

    # Calculate score
    print("0 error:  " + str(exprate.computeZero() * 100) + "%")
    print("1 error:  " + str(exprate.computeOne() * 100) + "%")
    print("<5 error: " + str(exprate.computeFive() * 100) + "%")
    print("<10 error:" + str(exprate.computeTen() * 100) + "%")
    




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
    model = transformer_vtex.Transformer(device, trg_vocab_size, trg_pad_idx, max_trg_length).to(device)
    exprate = test(model, device)