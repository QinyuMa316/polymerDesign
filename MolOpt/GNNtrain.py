
import time
import gc
import os
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# from MolOpt.utils import log_message, set_seed

# def train(model, train_loader, val_loader, optimizer, criterion, device,
#           epochs=50, patience=10, model_path='best_model.pth',
#           resume=True, initial_epoch=0):
#     model.to(device)
#     best_val_loss = float('inf')
#     no_improve = 0
#
#     if resume and os.path.exists(model_path):
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         best_val_loss = checkpoint['best_val_loss']
#         initial_epoch = checkpoint['epoch']
#         log_message(f"Resuming training from epoch {initial_epoch + 1}, best val loss so far: {best_val_loss:.4f}")
#
#     for epoch in range(initial_epoch, epochs):
#         start_time = time.time()
#         model.train()
#         train_loss = 0.0
#
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             out, _ = model(batch)
#             loss = criterion(out, batch.y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * batch.num_graphs
#
#             # 及时释放内存
#             del batch, out, loss
#             torch.cuda.empty_cache()
#
#         train_loss /= len(train_loader.dataset)
#
#         val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
#         end_time = time.time()
#         duration = (end_time - start_time) / 60
#         # 手动垃圾回收
#         gc.collect()
#         torch.cuda.empty_cache()
#
#         log_message(f'Epoch: {epoch + 1:03d}, '
#                     f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}, '
#                     f'Duration: {duration:.2f} min')
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             no_improve = 0
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_val_loss': best_val_loss,
#             }, model_path)
#         else:
#             no_improve += 1
#             if no_improve >= patience:
#                 log_message(f'Early stopping at epoch {epoch + 1}')
#                 break
#
#     # checkpoint = torch.load(model_path)
#     # model.load_state_dict(checkpoint['model_state_dict'])
#     model.load_state_dict(torch.load(model_path)['model_state_dict'])
#     return model

def train(model, train_loader, val_loader, optimizer, criterion, device,
          epochs=50, patience=10, model_path='best_model.pth',
          resume=True, initial_epoch=0,
          contrastive = False,
          ):
    # model.to(device)
    best_val_loss = float('inf')
    no_improve = 0
    if resume and os.path.exists(model_path):
        print(f'Successfully loaded model from {model_path}')
        # checkpoint = torch.load(model_path)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        if contrastive:
            train_loss = checkpoint['train_loss']
            print(f"Resuming training from epoch {initial_epoch + 1}, train loss so far: {train_loss:.4f}")
        else:
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming training from epoch {initial_epoch + 1}, best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"Failed to load model from {model_path}, starting training from scratch ...")


    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if contrastive:
                # 对比学习训练
                viewi, viewj = batch
                viewi = viewi.to(device)
                viewj = viewj.to(device)
                zi, readout = model(viewi, return_cl_dim = True)
                zj, readout = model(viewj, return_cl_dim = True)
                loss = criterion(zi, zj)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * viewi.num_graphs
            else:
                # 监督学习训练
                batch = batch.to(device)
                out, _ = model(batch, return_cl_dim = False)
                # pred = out, label = batch.y
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs

            # 及时释放内存
            # del batch, out, loss
            del batch, loss
            if not contrastive:
                del out
            torch.cuda.empty_cache()

        train_loss /= len(train_loader.dataset)

        # val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
        if contrastive:
            # 对比学习不验证
            val_loss, val_r2 = 0, 0
        else:
            val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
        end_time = time.time()
        duration = (end_time - start_time) / 60
        # 手动垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        if contrastive:
            print(f'Epoch: {epoch + 1:03d}, Train Loss: {train_loss:.6f}, Duration: {duration:.2f} min')
        else:
            print(f'Epoch: {epoch + 1:03d}, '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R2: {val_r2:.6f}, '
                  f'Duration: {duration:.2f} min')
        if contrastive:
            # 保存每个epoch的模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,  # 使用train_loss作为参考
            }, model_path)
            if epoch % 5 == 0:
                # 保存每5个epoch的模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,  # 使用train_loss作为参考
                }, model_path.replace('.pth', f'_{epoch}epoch.pth'))

        # if val_loss < best_val_loss:
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, model_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    # return model
    if not contrastive and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
            # 及时释放内存
            del batch, out, loss
            torch.cuda.empty_cache()

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    total_loss /= len(loader.dataset)
    r2 = r2_score(y_true, y_pred)

    # 手动垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    return total_loss, r2

def test(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out, _ = model(batch)
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

# def get_features(model, loader, device):
#     """获取所有样本的特征表示"""
#     model.eval()
#     features = []
#     labels = []
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             _, feat = model(batch)
#             features.append(feat.cpu().numpy())
#             labels.append(batch.y.cpu().numpy())
#
#     features = np.concatenate(features)
#     labels = np.concatenate(labels)
#     return features, labels


