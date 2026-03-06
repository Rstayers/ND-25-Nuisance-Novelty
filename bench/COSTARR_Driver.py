import torch
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import sys
import COSTARR 
import pdb
import argparse
import os
import cv2
import glob
import shutil
import torchvision as tv
from torchvision.models import ResNet50_Weights

###
# In this example, logits were extracted from a pretrained Big Transfer (https://github.com/google-research/big_transfer) 
# model using the ILSVRC_2012 dataset and 4 OOD datasets (SUN/iNaturalist/Textures/Places). 
#
# Files were saved as numpy arrays in the following format:
# [[Ground_Truth, logit_0, logit_1, ... logit_999],
# [Ground_Truth, logit_0, logit_1, ... logit_999],
# [Ground_Truth, logit_0, logit_1, ... logit_999]]
#
# Where the zeroth dimension/axis corresponds to each sample

# In this example we also have the option to use feature vector magnitudes as scalars for each sample in each dataset.
# The feature files are formatted just like logits where column 0 ([:,0]) is the ground truth
# and the zeroth dimension/axis corresponds to each sample

extractions_folder = "/home/rrabinow/ImageNet_OSR_OOD_Extractions/"
#extractions_folder = "/scratch/scruz/SPRING2023/ReMax_Extractions/"

architectures = []
ext_files = glob.glob(os.path.join(extractions_folder, "*"))
for path in ext_files:
    if '.npy' not in path or 'val_logits' not in path:
        continue
    filename = path.split("/")[-1]
    architectures.append(filename.replace("_val_logits.npy", ""))
    

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, required=True, choices=architectures, help="Underlying architecture")
parser.add_argument("--sys", type=str, required=True, choices=list(COSTARR.choices.keys()), help="Normalization to use")


args = parser.parse_args()

print("Start")


COSTARR_class = COSTARR.choices[args.sys]


debug = False
arch = args.arch
top_save_dir = 'COSTARR_Runs/'
if not os.path.exists(top_save_dir):
    os.mkdir(top_save_dir)

run_folder = top_save_dir +arch+"_"+args.sys+"/"

#Load the weights
weights=None
if  hasattr(COSTARR_class, 'needs_weight'):
    if arch == "convnextv2_H":
        weight_file = torch.load("/home/rrabinow/ImageNet_OSR_OOD_Extractions/Big_Extractors/ConvNeXtV2/convnextv2_huge_1k_224_ema.pt")
        weights = weight_file['model']['head.weight']
    elif arch == "hiera_H":
        weight_file = torch.load("/home/rrabinow/ImageNet_OSR_OOD_Extractions/Big_Extractors/Hiera/hiera_huge_224.pth")
        weights = weight_file['model_state']['head.projection.weight']
    elif arch == "resnet50":
        weights = tv.models.resnet50(weights = ResNet50_Weights.DEFAULT).fc.weight.detach().contiguous()
    elif arch == "mae_H":
        weight_file = torch.load("/home/rrabinow/ImageNet_OSR_OOD_Extractions/Big_Extractors/mae/mae_finetuned_vit_huge.pth")
        weights = weight_file['model']['head.weight']
    elif arch == "convnext_L":
        weight_file = torch.load("/home/rrabinow/ImageNet_OSR_OOD_Extractions/ConvNeXt/convnext_large_1k_224_ema.pth")
        weights = weight_file['model']['head.weight']
    else:
        print("No weights available")
        exit()


if not os.path.exists(run_folder):
    os.mkdir(run_folder)

shutil.copy("COSTARR.py", os.path.join(run_folder, "COSTARR_bkp.py"))

print("Loading files")

training_path = os.path.join(extractions_folder,arch+"_train_logits.npy")
training_FV_path = os.path.join(extractions_folder,arch+"_train_FV.npy")


training_data = torch.from_numpy(np.load(training_path))
training_FV_data = torch.from_numpy(np.load(training_FV_path))

#Ensure data is in the same order between FV and logit files!
assert torch.all(training_data[:,0] == training_FV_data[:,0]) == True

print("Filtering data")

#Separate GT from logit data
training_gt = training_data[:,0]

#Separate logits from GT
training_logits = training_data[:,1:]

#Separate FV from GT
training_FV = training_FV_data[:,1:]

#Filter the training samples so we only use samples where max_logit == GT
filtered_train_mask = training_gt == torch.max(training_logits, dim=1).indices
filtered_train_logits = training_logits[filtered_train_mask]
filtered_train_FV = training_FV[filtered_train_mask]
filtered_train_gt = training_gt[filtered_train_mask]


#################
#Fitting/Training
#################


if weights is None:
    COSTARR_model =  COSTARR_class(filtered_train_logits, filtered_train_FV)
else:
    print("Weighted")
    COSTARR_model =  COSTARR_class(filtered_train_logits, filtered_train_FV, weights)
##########
#Inference
##########
plots = []


datasets = ['val', 'v2_top', 'SUN', 'iNaturalist', 'Places', 'Textures', 'OpenImage_O', 'NINCO', 'easy_i21k', 'hard_i21k']

for dataset in datasets:
    print("dataset is set to "+ dataset)
    
    test_path = os.path.join(extractions_folder,arch+"_"+dataset+"_logits.npy")
    test_FV_path = os.path.join(extractions_folder,arch+"_"+dataset+"_FV.npy")
    
    
    test_data = torch.from_numpy(np.load(test_path))
    test_FV_data = torch.from_numpy(np.load(test_FV_path))
    

    test_gt = test_data[:,0] #Separate out GT
    test_logits = test_data[:,1:]
    test_FVs = test_FV_data[:,1:]
    test_preds = torch.max(test_logits, dim=1).indices
    
    if not 'val' in dataset and "v2" not in dataset: #If we're using an OOD dataset, force GT to be -1
        print(f'{dataset} is being treated as OOD')
        test_gt = (test_gt * 0) - 1
    
    
    model_outputs = COSTARR_model.ReScore(test_logits, test_FVs)
    if type(model_outputs) == tuple:
        assert len(model_outputs) == 2
        probs, max_index_test = model_outputs
    else:
        probs, max_index_test = torch.max(model_outputs, dim=1)
    
    
    #Save gt, pred, prob
    test_preds = torch.cat((test_gt.view(-1,1), max_index_test.view(-1,1), probs.view(-1,1)), dim=1)
    
    np.save(os.path.join(run_folder,f'COSTARR_{dataset}_preds.npy'), arr=test_preds.numpy())
    

sys.exit()

