import argparse
from utils import *
from train import *
from models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser("myDEGNN's Interface")
    parser.add_argument("--dataset" , type=str , default="celegans_small")
    parser.add_argument("--test_ratio" , type=float , default=0.2)
    parser.add_argument("--model" , type=str , default="DE-GNN")
    parser.add_argument("--seed" , type=float , default=0)
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    G , labels , task = read_file(args)
    print(labels)
    # exit(0)
    train_loader , val_loader , test_loader , feature_out = get_data(G , labels , task , args)
    model = GNNModel(in_features = train_loader.dataset[0].x.shape[-1] , 
                     hidden_features=128 , 
                     out_features = feature_out , 
                     model_name  = "DE-GNN")
    train_model(train_loader , model)