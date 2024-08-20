import imageio
import torch
import glob
import numpy as np

class Image_Dataset:
    def __init__(self, path, dstype = "train", imagesize = (360,360)):
        self.path = path
        self.dstype = dstype
        self.imagesize = imagesize
        self.fns = []
        self.celltypes = []
        self.ucells = []
        self.folders = glob.glob(f"{path}/*")
        for foldername in self.folders:
            with open(f"{foldername}/{dstype}_fns.txt", "r") as f:
                for line in f:
                    line = line.strip("\n")
                    self.fns.append(line)
                    endindex = 0
                    for i in range(5,len(line)):
                        if(line[i] == "/"):
                            endindex = i
                            break
                    celltype = line[5:endindex]
                    self.celltypes.append(celltype)
                    if(celltype not in self.ucells): #handle unique cell types
                        self.ucells.append(celltype)  
                    
    def __len__(self):
        return len(self.fns)
    def __getitem__(self, index):
        fn = self.fns[index]
        celltype = self.celltypes[index]
        image = imageio.v2.imread(fn)
        image = torch.from_numpy(image/255.0) 
        image = image.permute(2,0,1)
        image = torch.nn.functional.interpolate(image[None],size = self.imagesize)[0].float()
        for i in range(len(self.ucells)):
            if(celltype == self.ucells[i]):
                return image, torch.tensor(i, dtype = torch.int64)
    
