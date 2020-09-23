from torchvision import transforms
import torch.nn as nn
import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = "./minist_img"
path1 = "./parms/net.pt"
path2 = "./parms/arc.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,3,stride=2,padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),

        )
        self.liner=nn.Sequential(
            nn.Linear(32*14*14, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256 , 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2,bias=False)
        )

    def forward(self, x):
        conv=self.conv(x).view(-1,32*14*14)
        liner=self.liner(conv)
        return liner

class Arc(nn.Module):
    def __init__(self,feature_dim=2,cls_dim=10):
        super().__init__()
        self.W=nn.Parameter(torch.randn(feature_dim,cls_dim))

    def forward(self, feature,m=1,s=10):
        x=F.normalize(feature,dim=1)#x/||x||
        w = F.normalize(self.W, dim=0)#w/||w||
        cos = torch.matmul(x, w)/10
        # print(cos)
        # print(x)
        # print(w)
        a=torch.acos(cos)
        top=torch.exp(s*torch.cos(a+m))
        down2=torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))
        # print(a)
        # print(down2)
        out=torch.log(top/(top+down2))
        return out
net = Net().to(device)
if os.path.exists(path1):
    net = torch.load(path1)
arc = Arc(2,10).to(device)
if os.path.exists(path1):
    arc = torch.load(path2)
transform = transforms.Compose(
    [
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=(0.485, 0.456, 0.406),
        #     std=(0.229, 0.224, 0.225))
    ])
def compare(face1,face2):
    face1_norm = nn.functional.normalize(face1,dim=0)
    face2_norm = nn.functional.normalize(face2,dim=0)
    casa = torch.matmul(face1_norm,face2_norm.t())
    return casa
for file_path in os.listdir(img_path):
    i = 0
    feat_list = []
    img_name = ["img1","img2","img3"]
    fil_pth = os.path.join(img_path,file_path)
    for file in os.listdir(fil_pth):
        path = os.path.join(fil_pth,file)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)


        img_name[i] = img.copy()
        tran = transforms.ToTensor()

        im = tran(img).to(device)

        im = torch.unsqueeze(im,0).to(device)
        net.eval()
        feat = net(im)
        #out = arc(feat)
        feat_list.append(feat)
        i+=1

    fe = torch.cat(feat_list, 0)
    a = compare(fe[0], fe[1]).item()
    b = compare(fe[0], fe[2]).item()
    print(a, b)
    imstack =np.hstack((img_name[0],img_name[1],img_name[2]))
    cv2.imshow("imstack",imstack)
    cv2.waitKey(2000)
cv2.destroyAllWindows()