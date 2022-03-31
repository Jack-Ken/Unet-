import torch as t
import numpy as np
import PIL.Image as Image
import io, gc
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from modeals import UnetModeal
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Data import dataset
import matplotlib as plt

PATH = "./modeals/unet_model.pth"

# device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0")

x_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

y_transform = transforms.ToTensor()

def train_model(model, optimizer, criterion, dataloader, num_epochs = 10):
    """

    :param model:
    :param optimizer:
    :param criterion:
    :param dataloader:
    :param num_epochs:
    :return:
    """
    best_model = model
    min_loss = 1000
    model = model.to(device)
    for epoch in range(num_epochs):
        print("epoch{}".format(epoch))
        # dt_size = len(dataloader.dataset)
        epoch_loss = 0
        step = 0
        print("1111111")
        for index, (x, y) in enumerate(dataloader):
        # for x, y in tqdm(dataloader):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch {} loss:{}".format(epoch, epoch_loss/float(step)))
        if epoch_loss < min_loss:
            min_loss = epoch_loss / float(step)
            best_model = model
        gc.collect()
        t.cuda.empty_cache()
    # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    # t.save(state, PATH)
    t.save(best_model.state_dict(), PATH)
    return best_model


def train():
    """
    训练函数
    :return:
    """
    model = UnetModeal.Unet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    train_dataset = dataset.TrainDataset("./dataset/train/image/", "./dataset/train/label/", transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(train_dataset, batch_size=1)
    train_model(model, optimizer, criterion, dataloader)


def test():
    """
    模型测试
    :return:
    """
    modeal = UnetModeal.Unet()
    modeal.load_state_dict(t.load(PATH))
    test_dataset = dataset.TestDataset("./dataset/test", transform=x_transform)
    dataloader = DataLoader(test_dataset, batch_size=1)
    modeal.to(device)
    modeal.eval()
    plt.ion()
    with t.no_grad():
        for index, x in enumerate(dataloader):
            x.to(device)
            y = modeal(x)
            y.cpu()
            img_y = t.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]
            img = img_y[:, :, 0]
            img = np.interp(img, (img.min(), img.max()), (0, 255))
            im = Image.fromarray(img)
            if im.mode == "F":
                im = im.convert('L')
            im.save("./predict/" + str(index) + "_predict.png")
            # io.imsave("./dataset/predict/" + str(index) + "_predict.png", img)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    gc.collect()
    t.cuda.empty_cache()
    print("训练开始")
    train()
    print("训练完成，保存模型")
    print("-" * 20)
    print("开始预测")
    test()


