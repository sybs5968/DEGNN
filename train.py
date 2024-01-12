from param import *

def train_model(train_loader , model):
    epochs = default_epochs
    optimizer = torch.optim.SGD(model.parameters() , lr = default_lr , weight_decay = default_wd)
    for epoch in range(epochs):
        optimize_model(model, train_loader, optimizer)
        train_loss, train_acc, train_auc = eval_model(model, train_loader)
        # model.train()
        # for batch in train_loader:
        #     label = batch.y
        #     predi = model(batch)
        #     loss = torch.nn.functional.cross_entropy(predi , label , reduction='mean')
        #     loss.backward()
        #     optimizer.step()

        # model.eval()
        # labels = []
        # predictions = []
        # with torch.no_grad():
        #     for batch in train_loader:
        #         labels.append(batch.y)
        #         predictions.append(model(batch))
        #     labels = torch.cat(labels , dim = 0)
        #     predictions = torch.cat(predictions , dim = 0)
        #     train_loss , train_acc , train_auc = compute_metric(predictions , labels)
            # # print(labels.shape)
            # # print(predictions.shape)
            # train_loss = torch.nn.functional.cross_entropy(predictions , labels , reduction='mean').item()
            # correct = (torch.argmax(predictions , dim = 1) == labels)
            # train_acc = correct.sum().cpu().item() / labels.shape[0]
            # predictions = torch.nn.functional.softmax(predictions , dim = -1)
            # multi_class = "ovr"
            # if predictions.size(1) == 2:
            #     predictions = predictions[: , 1]
            #     multi_class = "raise"
            # # print(labels.shape)
            # # print(predictions.shape)
            # print(multi_class)
            # train_auc = 0
            # # train_auc = roc_auc_score(labels.cpu().numpy() , predictions.cpu().numpy() , multi_class=multi_class)
        print("%d: loss = %lf , acc = %lf , auc = %lf" % (epoch , train_loss , train_acc , train_auc))

def optimize_model(model, dataloader, optimizer):
    model.train()
    # setting of data shuffling move to dataloader creation
    for batch in dataloader:
        # batch = batch.to(device)
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
            # batch = batch.to(device)
            labels.append(batch.y)
            # print(labels[-1])
            # print(batch.y.shape)
            prediction = model(batch)
            # print(prediction.shape)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        # print(labels)
        # exit(0)
    if not return_predictions:
        loss, acc, auc = compute_metric(predictions, labels)
        return loss, acc, auc
    else:
        return predictions

def compute_metric(predictions, labels):
    # print(predictions.shape)
    # print(labels.shape)
    with torch.no_grad():
        # compute loss:
        loss = torch.nn.functional.cross_entropy(predictions, labels, reduction='mean').item()
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        # compute auc:
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        # print(labels.shape)
        # print(predictions.shape)
        # print(multi_class)
        # print(labels)
        # print(predictions)
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc