import torchvision.models as models
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        ''' declare layers used in this network '''
        pretrained_model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(
            *list(pretrained_model.children())[:-2],
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, img):
        return self.model(img)
