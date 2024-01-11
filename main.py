from utils import *
from train import *
from models import *

if __name__ == "__main__":
    G , labels = read_file()
    train_loader , val_loader , test_loader , feature_out = get_data(G , labels)
    print(train_loader.dataset[0].x.shape)
    model = GNNModel(in_features = train_loader.dataset[0].x.shape[-1] , hidden_features=128 , out_features = feature_out , model_name  = "DE-GNN")
    train_model(train_loader , model)