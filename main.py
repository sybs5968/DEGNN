import os
import sys
import argparse
from utils import *
from train import *
from models import *
from datetime import datetime


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("myDEGNN's Interface")
    parser.add_argument("--lr"              , type=float , default=1e-4)
    parser.add_argument("--wd"              , type=float , default=0)
    parser.add_argument("--seed"            , type=float , default=0)
    parser.add_argument("--model"           , type=str   , default="DE-GNN")
    parser.add_argument("--epoch"           , type=int   , default=100)
    parser.add_argument("--layers"          , type=int   , default=2)
    parser.add_argument("--dropout"         , type=float , default=0)
    parser.add_argument("--dataset"         , type=str   , default="celegans_small")
    parser.add_argument("--feature"         , type=str   , default="sp")
    parser.add_argument("--test_ratio"      , type=float , default=0.2)
    parser.add_argument("--prop_depth"      , type=int   , default=1)
    parser.add_argument("--hidden_features" , type=int   , default=128)

    parser.add_argument("--feature_limit"   , type=int   , default=3)

    parser.add_argument("--log_path"        , type=str   , default="./log")
    parser.add_argument("--metric"          , type=str   , default="acc")
    args = parser.parse_args()
    error_massgae = check(args)
    if error_massgae:
        print("Error: " + error_massgae)
        exit(0)
    log_path = os.path.join(args.log_path , args.dataset)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    log_filename = datetime.now().strftime("%m-%d_%H:%M:%S") + '_' + args.feature + '_' + str(args.epoch) + ".txt"

    logger = open(log_path + '/' + log_filename , "w")

    set_random_seed(args.seed)

    G , labels , task = read_file(args)
    print("YES ONE")
    train_loader , val_loader , test_loader , feature_out = get_data(G , labels , task , args)
    print("YES TOW")
    model = GNNModel(in_features = train_loader.dataset[0].x.shape[-1] , hidden_features=128 , out_features = feature_out , model_name  = args.model)
    print("YES THREE")
    train_model(train_loader , val_loader , test_loader , model , args , logger)