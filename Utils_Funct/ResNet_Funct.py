# ResNet_Funct
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F


def resnet_training(
        net, 
        train_dataloader, valid_dataloader, 
        criterion, optimizer, 
        scheduler=None, epochs:int=10, 
        device:str='cpu', model_name:str='our_resnet'
):

    # Initialize lists to store losses and accuracies
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    best_valid_loss = float('inf')

    for epoch in range(1,epochs+1):
        start_time = time.time()

        net.train()
        train_loss = torch.tensor(0., device=device)
        train_accuracy = torch.tensor(0., device=device)

        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            preds = net(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_accuracy += (torch.argmax(preds, dim=1) == y).sum()

        train_losses.append(train_loss/len(train_dataloader.dataset))
        train_accs.append(100*train_accuracy/len(train_dataloader.dataset))


        if valid_dataloader is not None:
            net.eval() 
            valid_loss = torch.tensor(0., device=device)
            valid_accuracy = torch.tensor(0., device=device)

            with torch.no_grad():
                for X, y in valid_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    preds = net(X)
                    loss = criterion(preds, y)

                    valid_loss += loss * valid_dataloader.batch_size
                    valid_accuracy += (torch.argmax(preds, dim=1) == y).sum()

        end_time = time.time()
        valid_losses.append(valid_loss/len(valid_dataloader.dataset))
        valid_accs.append(100*valid_accuracy/len(valid_dataloader.dataset))

        if scheduler is not None:
            scheduler.step()

        print(f"\nEpoch: {epoch}/{epochs} -- Epoch Time: {end_time-start_time:.2f} s")
        print("---------------------------------")
        print(f"Train -- Loss: {train_loss/len(train_dataloader.dataset):.3f}, Acc: {train_accs[-1]:.2f}%")
        print(f"Val -- Loss: {valid_loss/len(valid_dataloader.dataset):.3f}, Acc: {valid_accs[-1]:.2f}%")

        if valid_losses[-1] < best_valid_loss:
            best_valid_loss = valid_losses[-1]
            torch.save(net.state_dict(), model_name)

    return net, train_losses, train_accs, valid_losses, valid_accs


def plot_resnet_results(n_epochs, train_losses, train_accs, valid_losses, valid_accs):
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(n_epochs) + 1, train_losses, linewidth=3, label='Train Loss')
    plt.plot(np.arange(n_epochs) + 1, valid_losses, linewidth=3, label='Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during training')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(n_epochs) + 1, train_accs, linewidth=3, label='Train Accuracy')
    plt.plot(np.arange(n_epochs) + 1, valid_accs, linewidth=3, label='Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy during training')

    plt.show()


def plot_resnet_confusion_matrix(model, dataloader, true_labels, class_names=None, device='cuda'):
    # Ensure model is in evaluation mode
    model.eval()

    preds = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            preds.extend(model(X).argmax(dim=1).cpu().numpy())  # Get predicted labels

    # Convert predictions to numpy array
    preds = np.array(preds)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, preds)

    # Print classification report
    print("\nClassification Report:\n", classification_report(true_labels, preds))

    if class_names is None:
        class_names = [str(i) for i in np.unique(true_labels)] 
        
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()




###################################################################################################

def resnet_train_sr(net, feature_extractor1, feature_extractor2, train_dataloader, valid_dataloader,
                 criterion, optimizer, scheduler=None, epochs=10, device='cpu', model_name:str="best_resnet_sr.pt"):

    # Initialize lists to store losses and accuracies
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start = time.time()

        feature_extractor1.train()
        feature_extractor2.train()
        net.train()

        train_loss = torch.tensor(0., device=device)
        train_accuracy = torch.tensor(0., device=device)

        for X1, X2, y in train_dataloader:
            X1, X2 = X1.to(device), X2.to(device)
            y = y.to(device)

            # Use separate feature extractors
            X1 = feature_extractor1(X1)
            X2 = feature_extractor2(X2)
            X = torch.sub(X1, X2)

            preds = net(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_accuracy += (torch.argmax(preds, dim=1) == y).sum()

        train_loss = train_loss / len(train_dataloader.dataset)
        train_accuracy = 100 * train_accuracy / len(train_dataloader.dataset)

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        if valid_dataloader is not None:
            net.eval()  # put network in eval mode
            valid_loss = torch.tensor(0., device=device)
            valid_accuracy = torch.tensor(0., device=device)

            with torch.no_grad():
                for X1, X2, y in valid_dataloader:
                    X1, X2 = X1.to(device), X2.to(device)
                    y = y.to(device)

                    # Use separate feature extractors
                    X1 = feature_extractor1(X1)
                    X2 = feature_extractor2(X2)
                    X = torch.sub(X1, X2)

                    preds = net(X)
                    loss = criterion(preds, y)

                    valid_loss += loss * valid_dataloader.batch_size
                    valid_accuracy += (torch.argmax(preds, dim=1) == y).sum()
        
        val_accuracy = 100 * valid_accuracy / len(valid_dataloader.dataset)
        val_loss = valid_loss / len(valid_dataloader.dataset)

        valid_losses.append(val_loss)
        valid_accs.append(val_accuracy)

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(net.state_dict(), model_name)
            torch.save(feature_extractor1.state_dict(), model_name.split(".pt")[0]+"_f_extr1.pt")
            torch.save(feature_extractor2.state_dict(), model_name.split(".pt")[0]+"_f_extr2.pt")
        
        end = time.time()
        
        print(f"\nEpoch: {epoch+1}/{epochs} -- Epoch Time: {end-start:.2f} s")
        print("---------------------------------")
        print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_accuracy:.2f}%")
        print(f"Val -- Loss: {val_loss:.3f}, Acc: {val_accuracy:.2f}%")


    
    print(f'Total training time: {end - start:.1f} seconds')
    return train_losses, train_accs, valid_losses, valid_accs


def plot_resnet_confusion_matrix_sr(model, f_extr1, f_extr2, dataloader, class_names=None, device='cuda'):
    model.eval()

    true_labels = []
    preds = []

    with torch.no_grad():
        for rt1, pt1, y in dataloader: 
            rt1, pt1, y = rt1.to(device), pt1.to(device), y.to(device)

            rt1 = f_extr1(rt1)
            pt1 = f_extr2(pt1)

            model_input = torch.sub(rt1, pt1)

            y_pred = model(model_input)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1) 
            preds.append(top_pred)
            true_labels.append(y)

    preds = torch.cat(preds).cpu().numpy().flatten()
    true_labels = torch.cat(true_labels).cpu().numpy().flatten()

    cm = confusion_matrix(true_labels, preds)

    print("\nClassification Report:\n", classification_report(true_labels, preds))

    if class_names is None:
        class_names = [str(i) for i in np.unique(true_labels)] 
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()














