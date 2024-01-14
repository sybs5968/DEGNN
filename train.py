from param import *
from utils import logINFO
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_model(train_loader , val_loader , test_loader , model , args , logger):
    epochs = args.epoch
    loss = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters() , lr = args.lr , weight_decay = args.wd)
    MaxAnswer = 0
    animator = Animator(xlabel="epoch" , ylabel=args.metric , xlim=[1 , epochs])
    for epoch in range(epochs):
        optimize_model(model, train_loader, optimizer , loss)
        train_loss, train_acc, train_auc = eval_model(model, train_loader)
        val_loss  , val_acc  , val_auc   = eval_model(model , val_loader)
        test_loss , test_acc , test_auc  = eval_model(model , test_loader)

        if args.metric == "acc":
            MaxAnswer = max(MaxAnswer , test_acc)
            Message = "epoch %4d best test acc: %.4lf, train loss: %.4lf; train acc: %.4lf, val acc: %.4lf, test acc: %.4lf" % (epoch , MaxAnswer , train_loss , train_acc , val_acc , test_acc)
        else:
            MaxAnswer = max(MaxAnswer , test_auc)
            Message = "epoch %4d best test auc: %.4lf, train loss: %.4lf; train auc: %.4lf, val auc: %.4lf, test auc: %.4lf" % (epoch , MaxAnswer , train_loss , train_auc , val_auc , test_auc)
        animator.add(epoch + 1 , MaxAnswer)
        logINFO(Message , logger)
        # print(Message)

def optimize_model(model, dataloader, optimizer , loss):
    model.train()
    for batch in dataloader:
        label = batch.y
        prediction = model(batch)
        loss_val = loss(prediction, label, reduction='mean')
        loss_val.backward()
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