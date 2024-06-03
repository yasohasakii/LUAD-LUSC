import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import transformer, network, losses
from torch.utils import data


# Modified on @Yunfan Li's Contrastive-Clustering code. https://github.com/Yunfan-Li/Contrastive-Clustering

def save_model(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)

if __name__ == "__main__":
    model_path = 'pretrained_models'
    seed = 1
    dataset_dir = r'D:\workspace\LUAD&LUSC\data\cancer\jpg\*'
    image_size = 256
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.1
    start_epoch = 0
    epochs = 50
    workers = 4
    reload = False
    instance_temperature = 0.1
    cluster_temperature = 0.1
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # prepare data
    train_dataset = transformer.CLSDataset(
        root=dataset_dir,
        transform=transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(p=0.33),
                                      transforms.RandomVerticalFlip(p=0.33),
                                      transforms.RandomRotation(degrees=10),
                                      transforms.ToTensor()])
    )
    test_dataset = transformer.CLSDataset(
        root=dataset_dir,
        transform=transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),])
    )
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
    )
    class_num = 5

    # initialize model
    model = network.LUCLSModel()
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if reload:
        model_fp = os.path.join(model_path, "checkpoint_{}.pt".format(start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    device = torch.device("cuda")
    criterion= nn.CrossEntropyLoss().to(device)
    # train
    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = 0
        for step, (x, y) in enumerate(data_loader):
            x = x.to('cuda')
            y = y.to('cuda')
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(
                    f"Step [{step}/{len(data_loader)}]\t loss: {loss.item()}")
            loss_epoch += loss.item()
        if epoch % 5 == 0:
            save_model(model_path, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(model_path, model, optimizer, epochs)
