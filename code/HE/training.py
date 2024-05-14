import os
import numpy as np
import torch
import torchvision
import transformer, network, losses
from torch.utils import data


# Modified on @Yunfan Li's Contrastive-Clustering code. https://github.com/Yunfan-Li/Contrastive-Clustering
def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch

def save_model(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)

if __name__ == "__main__":
    model_path = ''
    seed = 1
    dataset_dir = ''
    image_size = 256
    batch_size = 4
    learning_rate = 1e-3
    weight_decay = 0.1
    start_epoch = 0
    epochs = 500
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
    train_dataset = transformer.CIDataset(
        root=dataset_dir,
        transform=transformer.Transforms( s=0.5),
    )
    test_dataset = transformer.CIDataset(
        root=dataset_dir,
        transform=transformer.Transforms( s=0.5),
    )
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
    )
    class_num = 2

    # initialize model
    model = network.LUCCModel()
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if reload:
        model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    loss_device = torch.device("cuda")
    criterion_instance = losses.InstanceLoss(batch_size, instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = losses.ClusterLoss(class_num,cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(model_path, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(model_path, model, optimizer, epochs)
