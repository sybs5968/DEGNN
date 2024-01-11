from param import *

def train_model(train_loader , model):
    epochs = default_epochs
    optimizer = torch.optim.SGD(model.parameters() , lr = default_lr , weight_decay = default_wd)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            label = batch.y
            predi = model(batch)
            loss = torch.nn.functional.cross_entropy(predi , label , reduction='mean')
            loss.backward()
            optimizer.step()

        model.eval()
        labels = []
        predictions = []
        with torch.no_grad():
            for batch in train_loader:
                labels.append(batch.y)
                predictions.append(model(batch))
            labels = torch.cat(labels , dim = 0)
            predictions = torch.cat(predictions , dim = 0)
            train_loss = torch.nn.functional.cross_entropy(predictions , labels , reduction='mean').item()
            correct = (torch.argmax(predictions , dim = 1) == labels)
            train_acc = correct.sum().cpu().item() / labels.shape[0]
            predictions = torch.nn.functional.softmax(predictions , dim = -1)
            multi_class = "ovr"
            if predictions.size(1) == 2:
                predictions = predictions[: , 1]
                multi_class = "raise"
            train_auc = roc_auc_score(labels.cpu().numpy() , predictions.cpu().numpy() , multi_class=multi_class)
        print("%d: loss = %lf , acc = %lf , auc = %lf" % epoch , train_loss , train_acc , train_auc)