import os
import torch
import numpy as np
import sys
import pdb
from tqdm import tqdm
import math
import scipy

n_cpu = int(os.cpu_count()*0.8)

class Postprocessor_Base:
    #Data must be a 2-D tensor containing the logits of correctly classified samples (where dim-0 is samples, dim-1 is vectors)
    #Optionally, rescale can be a 1-D tensor of scalar values associated with each sample in data
    def __init__(self, data, FVs): 
        print("init")
    
    #Data must be a 2-D tensor containing the logits of all test time samples
    def ReScore(self, data, FVs):

        normd_data = self.norm(data, FVs)
        return normd_data
    
    def norm(self, data, FVs):
        l2 = torch.norm(FVs, p=2, dim=1).view(-1,1)
        return data/l2



    
class COSTARR(Postprocessor_Base):
    needs_weight=True
    def __init__(self, data, FVs, weights):
        # Ensure weights are on the same device as feature vectors
        self.device = FVs.device
        self.weights = weights.to(self.device)
        self.setup_dict = self.setup(data, FVs)
        Postprocessor_Base.__init__(self, data, FVs)
        return
        
        
    def norm(self, logits, FVs):
        pred = torch.max(logits, dim=1).indices
        device = FVs.device
        normalized_logits = torch.zeros(logits.shape, device=device)

        print("Iterating classes for COSTARR scoring")
        for c in tqdm(self.setup_dict.keys()):
            class_mask = pred == c
            mean, hmean = self.setup_dict[c]

            cat_FV = torch.cat([FVs[class_mask],FVs[class_mask]*self.weights[c]], dim=1)

            cat_mean = torch.cat([mean, hmean], dim=0)

            scaled_class_logits = torch.minimum(torch.ones(logits[class_mask, c].shape, device=device),torch.maximum(torch.zeros(logits[class_mask, c].shape, device=device), (logits[class_mask, c]-self.logit_min)/(self.logit_max-self.logit_min)))
            
            
            COSTARR = (1+torch.nn.functional.cosine_similarity(cat_FV, cat_mean))/2
            normalized_logits[class_mask,c] = scaled_class_logits*COSTARR

        return normalized_logits

    def setup(self, logits, FV): #ASSUME DATA IS CORRECTLY PREDICTED
        preds = torch.max(logits, dim=1).indices
        classes = torch.unique(preds).long().tolist()     
        
        class_models = {}
        print("Generating models")
        
        
        self.logit_min = torch.min(logits)
        self.logit_max = torch.max(logits)
        
        print(f"Min:{self.logit_min}, Max:{self.logit_max}")
        
        for c in tqdm(classes):
            select_class_FVs = FV[preds == c]
            hmean = torch.mean(select_class_FVs*self.weights[c], dim=0)
            mean = torch.mean(select_class_FVs, dim=0)
            class_models[c] = (mean, hmean)

        
        return class_models

choices = {

        "COSTARR":COSTARR
    }

if __name__ == '__main__':
    mp.set_start_method("fork", force=True)
    print("")


    