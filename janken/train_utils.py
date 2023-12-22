import torch

def training(model, train_loader, val_loader, criterion, optimizer, device):
    # Train loop ----------------------------
    model.train()  # 学習モードをオン
    train_batch_loss = []
    for data, label in train_loader:
        # GPUへの転送
        data, label = data.to(device), label.to(device)
        # 1. 勾配リセット
        optimizer.zero_grad()
        # 2. 推論
        output = model(data)
        # 3. 誤差計算
        loss = criterion(output, label)
        # 4. 誤差逆伝播
        loss.backward()
        # 5. パラメータ更新
        optimizer.step()
        # train_lossの取得
        train_batch_loss.append(loss.item())

    # val(val) loop ----------------------------
    model.eval()  # 学習モードをオフ
    val_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            val_batch_loss.append(loss.item())

    return model, train_batch_loss, val_batch_loss