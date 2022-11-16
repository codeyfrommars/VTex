
def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x = torch.tensor(batch[:, 0]).to(device)
        y = torch.tensor(batch[:, 1]).to(device)

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

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            y = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

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


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
    
    return train_loss_list, validation_loss_list
