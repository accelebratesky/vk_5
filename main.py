import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import build_loaders
from model import Mymodel


if __name__ == "__main__":
    data_root = r".\data"

    train_loader, val_loader, test_loader, classes = build_loaders(data_root=data_root, batch_size=64)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mymodel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_accuracy = []
    val_accuracy = []
    epochs = 30
    best_final_loss = float("inf")
    early_stop_patience = 12
    cnt = 0
    best_final_w = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_num = 0
        correct_train = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch:04d} Train", leave=False)
        for bx, by in pbar_train:
            bx, by = bx.to(device), by.to(device)
            prediction = model(bx)
            loss = loss_fn(prediction, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = bx.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_num += batch_size
            pred_label = torch.argmax(prediction, dim=1)
            correct_train += (pred_label == by).sum().item()


            pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / total_train_num
        train_acc = correct_train / total_train_num


        model.eval()
        total_val_loss = 0.0
        total_val_num = 0
        correct_val = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:04d} Val", leave=False)
        with torch.no_grad():
            for bx, by in pbar_val:
                bx, by = bx.to(device), by.to(device)
                prediction = model(bx)
                loss = loss_fn(prediction, by)
                batch_size = bx.size(0)
                total_val_loss += loss.item() * batch_size
                total_val_num += batch_size
                pred_label = torch.argmax(prediction, dim=1)
                correct_val += (pred_label == by).sum().item()
                pbar_val.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / total_val_num
        val_acc = correct_val / total_val_num

        print(
            f"epoch:{epoch:4d} | train_loss:{avg_train_loss:.4f} | train_acc:{train_acc:.4f} | val_loss:{avg_val_loss:.4f} | val_acc:{val_acc:.4f}")

        if avg_val_loss < best_final_loss:
            best_final_loss = avg_val_loss
            best_final_w = copy.deepcopy(model.state_dict())
            cnt = 0
        else:
            cnt += 1
            if cnt >= early_stop_patience:
                print(f"Early Stop at epoch {epoch}, best val loss: {best_final_loss:.4f}")
                break

        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)


    print("\n===== 加载最优权重，测试集评估 =====")
    model.load_state_dict(best_final_w)
    model.eval()
    total_test_loss = 0.0
    total_test_num = 0
    correct_test = 0
    pbar_test = tqdm(test_loader, desc="Test")
    with torch.no_grad():
        for bx, by in pbar_test:
            bx, by = bx.to(device), by.to(device)
            prediction = model(bx)
            loss = loss_fn(prediction, by)
            batch_size = bx.size(0)
            total_test_loss += loss.item() * batch_size
            total_test_num += batch_size
            pred_label = torch.argmax(prediction, dim=1)
            correct_test += (pred_label == by).sum().item()

    avg_test_loss = total_test_loss / total_test_num
    test_acc = correct_test / total_test_num
    print(f"\n==== Final Test Result ====")
    print(f"test_loss:{avg_test_loss:.4f} | test_acc:{test_acc:.4f}")

