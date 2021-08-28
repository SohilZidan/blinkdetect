import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def criterion(y_pred, y_true, log_vars):
    loss = 0
    for i in range(len(y_pred)):
        precision = torch.exp(-log_vars[i]).to(device)
        diff = (y_pred[i]-y_true[i])**2.
        loss += torch.sum(precision * diff + log_vars[i].to(device), -1)
    return torch.mean(loss)

def criterion_0(losses, log_vars):
    loss0 = losses[0]
    loss1 = losses[1]
    
    precision0 = torch.exp(-log_vars[0]).to(device)
    loss0 = precision0*loss0 + log_vars[0].to(device)

    precision1 = torch.exp(-log_vars[1]).to(device)
    loss1 = precision1*loss1 + log_vars[1].to(device)

    # print(loss0+loss1)
    
    return torch.sqrt(loss0**2+loss1**2)
    # return torch.mean(torch.tensor(losses))

def criterion_1(loss_val, log_vars):
    loss = 0
    for i in range(len(loss_val)):
        precision = torch.exp(-log_vars[i]).to(device)
        diff = loss_val[i]
        loss += torch.sum(precision * diff + log_vars[i].to(device), -1)
    return torch.mean(loss)