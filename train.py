from param import *

def train_model(train_loader , model):
    epochs = default_epochs
    optimizer = torch.optim.SGD(model.parameters() , lr = default_lr , weight_decay = default_wd)
    for epoch in range(epochs):
        optimize_model(model, train_loader, optimizer)
        train_loss, train_acc, train_auc = eval_model(model, train_loader)
        print("%d: loss = %lf , acc = %lf , auc = %lf" % (epoch , train_loss , train_acc , train_auc))

def optimize_model(model, dataloader, optimizer):
    model.train()
    # setting of data shuffling move to dataloader creation
    for batch in dataloader:
        label = batch.y
        prediction = model(batch)
        loss = torch.nn.functional.cross_entropy(prediction, label, reduction='mean')
        loss.backward()
        optimizer.step()

def eval_model(model, dataloader, return_predictions=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            labels.append(batch.y)
            prediction = model(batch)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
    if not return_predictions:
        loss, acc, auc = compute_metric(predictions, labels)
        return loss, acc, auc
    else:
        return predictions

def compute_metric(predictions, labels):
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(predictions, labels, reduction='mean').item()
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc