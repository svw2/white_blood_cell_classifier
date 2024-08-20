from data import *
from nwk import *
from torch.utils.data import DataLoader

tdata = Image_Dataset("data")
vdata = Image_Dataset("data", "valid")
tload = DataLoader(tdata, 8, True)
vload = DataLoader(vdata, 8)
network = NWK(len(tdata.unumbers),784) 
error_function = nn.NLLLoss()
proc = torch.optim.Adam(network.parameters(),lr=0.001)


for i in range(5):
    for image,label in tload:
        image = image.view(image.shape[0], image.shape[1], -1)
        p = network(image)
        error = error_function(p[:,0],label)
        error.backward()
        proc.step()


    print(f"i: {i+1}, error:{error}") 
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
    print(i)
    print(f"count: {count}, total: {total}")
    print((count/total)*100)
    network.train()
torch.save(network.state_dict(),"network.pth")
        

