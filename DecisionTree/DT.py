import torch
import math
from math import log2

def get_entropy_of_dataset(tensor:torch.Tensor):
    psum=0
    nsum=0

    lis=[i[-1] for i in tensor]
    psum=lis.count(1)
    nsum=lis.count(0)
    total=psum+nsum
    return -(psum/total*log2(psum/total)+nsum/total*log2(nsum/total))
    

def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
        target_attribute = tensor[:,attribute]

        size = target_attribute.size()[0]

        numberOfUnique = torch.unique(target_attribute)


        l = {}

        for i in numberOfUnique:
            row=0
            type_size=0
            type_yes=0
            for keys in target_attribute:
                if i==keys and tensor[row,-1]:
                    type_yes+=1
                if(i==keys):
                    type_size+=1
                row+=1
            l[i]=[type_yes,type_size]


        type_entropy=[]
        for attribute_type in l.values():
            p=attribute_type[0]/attribute_type[1]
            q=1-p
            if(p==1 or q==1 or p==0 or 1==0):
                entropy = 0
            else:
                entropy =-( p * math.log(p,2) + q * math.log(q,2)  )
            type_entropy.append((entropy,attribute_type[1]))

        weighted_avg=0;

        for i in type_entropy:
            if(i[0]==0):
                continue
        
            weighted_avg += (i[1]/size) * i[0]
        
        return weighted_avg


def get_information_gain(tensor:torch.Tensor, attribute:int):
    info_gain = get_entropy_of_dataset(tensor) - get_avg_info_of_attribute(tensor,attribute)
    return info_gain

def get_selected_attribute(tensor:torch.Tensor):
    columns = tensor.shape[1] - 1
    dict = {}
    for i in range(columns):
        dict[i] = get_information_gain(tensor,i)
    tuple_val = max(dict, key=dict.get)
    return dict,tuple_val