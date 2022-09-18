import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=output.dim()-1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        
    return correct / torch.prod(torch.tensor(target.size()))

def top_k_acc(output, target, k=2):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
