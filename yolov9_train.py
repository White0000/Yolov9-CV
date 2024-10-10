import os
import torch
from models.yolo import Model
from utils.datasets import LoadImagesAndLabels
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device


# 初始化 YOLOv9 模型并加载预训练模型
def load_yolov9_model(weights='yolov9_model/yolov9-c.pt', imgsz=640):
    device = select_device('0')  # 使用GPU设备
    model = Model()  # 初始化YOLOv9模型
    model.load_state_dict(torch.load(weights, map_location=device)['model'].state_dict())  # 加载预训练模型
    model.to(device).eval()  # 切换到推理模式
    return model, device, imgsz


# 数据加载器和训练准备
def load_data(data_path, img_size):
    dataset = LoadImagesAndLabels(data_path, img_size, augment=True)  # 数据增强
    return dataset


def train_yolov9(model, dataset, epochs=100, batch_size=16):
    device = select_device('cuda:0')
    model.train()

    # 损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_idx, (imgs, targets, paths, _) in enumerate(dataset):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 前向传播
            pred = model(imgs)
            loss = model.compute_loss(pred, targets)[0]

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(dataset)}, Loss: {loss.item()}')

    # 保存模型
    torch.save(model.state_dict(), 'yolov9_trained_model.pth')


if __name__ == "__main__":
    # 加载模型与数据
    yolov9_model, device, imgsz = load_yolov9_model()
    dataset = load_data('dataset/train', imgsz)

    # 开始训练
    train_yolov9(yolov9_model, dataset)
