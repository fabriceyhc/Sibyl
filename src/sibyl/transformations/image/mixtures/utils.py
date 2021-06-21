import torch

def find_value_idx(tensor, value):
    return torch.nonzero(tensor == value, as_tuple=False).view(-1)

def find_other_idx(tensor, value):
    return torch.nonzero(tensor != value, as_tuple=False).view(-1)

def get_confusion_matrix(model, dataloader, device, normalize=True):
    n_classes = len(dataloader.dataset.classes)
    confusion_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1                    
    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=0)
    return confusion_matrix

def get_most_confused_per_class(confusion_matrix):
    idx = torch.arange(len(confusion_matrix))
    cnf = confusion_matrix.fill_diagonal_(0).max(dim=1)[1]
    return torch.stack((idx, cnf)).T.tolist()

def get_k_most_confused_per_class(confusion_matrix, k):
    lbl = torch.arange(len(confusion_matrix))
    cnf = confusion_matrix.fill_diagonal_(0).topk(k, dim=1)[-1]
    return lbl, cnf