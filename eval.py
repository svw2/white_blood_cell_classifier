from nwk import *
from data import *
from torch.utils.data import DataLoader


vdata = Image_Dataset("data", "test")
vload = DataLoader(vdata, 8)
network = NWK(len(vdata.unumbers),784) 
network.load_state_dict(torch.load("network.pth"))
count = 0
total = 0
network.eval()
for image,label in vload:
    image = image.view(image.shape[0], image.shape[1], -1)
    p = network(image)
    p_index = torch.argmax(p,dim=-1)[:,0]
    for j in range(label.shape[0]):
        if(p_index[j] == label[j]):
            count+=1
        total+=1
print((count/total)*100)

        



