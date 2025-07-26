# OurModel_with_SandR_Funct
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def model_training_SR(
        model, 
        dataloader_training, dataloader_validation, 
        optimizer, criterion, 
        num_epochs:int=10, 
        step_size:int=5, gamma:float=0.5,
        model_name:str="ournet_sr.pt",
        ):
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_valid_loss = float('inf')

    #Every 5 epochs (step_size), the learning rate is multiplied by 0.5 (gamma).
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        start_time = time.time()

        correct_train = 0
        total_train = 0
        running_train_loss = 0.0

        for rt1, pt1, labels in dataloader_training:
            optimizer.zero_grad()
            outputs = model(rt1, pt1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_train_loss / len(dataloader_training)
        train_accuracy = 100 * correct_train / total_train

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for rt1, pt1, labels in dataloader_validation:
                outputs = model(rt1, pt1)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        valid_loss = running_val_loss / len(dataloader_validation)
        val_accuracy = 100 * correct_val / total_val
        
        val_losses.append(valid_loss)
        val_accuracies.append(val_accuracy)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)
        
        end_time = time.time()
        scheduler.step() # adjust LR

        print(f"\nEpoch: {epoch+1}/{num_epochs} -- Epoch Time: {end_time-start_time:.2f} s")
        print("---------------------------------")
        print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_accuracy:.2f}%")
        print(f"Val -- Loss: {valid_loss:.3f}, Acc: {val_accuracy:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_SR(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for rt1, pt1, labels in iterator:
            rt1, pt1, labels = rt1.to(device), pt1.to(device), labels.to(device)

            y_pred = model(rt1, pt1)  # Forward pass
            loss = criterion(y_pred, labels)  # Compute loss
            acc = calculate_accuracy(y_pred, labels)  # Compute accuracy

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def calculate_accuracy(y_pred, y):
  
  y_prob = F.softmax(y_pred, dim = -1)
  y_pred = y_pred.argmax(dim=1, keepdim = True)
  correct = y_pred.eq(y.view_as(y_pred)).sum()
  acc = correct.float()/y.shape[0]
  return acc

def model_testing_SR(model, test_iterator, criterion, device, model_name):
    model.load_state_dict(torch.load(model_name)) 
    test_loss, test_acc = evaluate_SR(model, test_iterator, criterion, device) 
    print(f"Test -- Loss: {test_loss:.3f}, Acc: {test_acc * 100:.2f} %")

def predict_SR(model, iterator, device):
    '''
    Run predictions on a dataset iterator
    '''
    model.eval()
    
    labels = []
    pred = []

    with torch.no_grad():
        for rt1, pt1, y in iterator: 
            rt1, pt1, y = rt1.to(device), pt1.to(device), y.to(device)

            y_pred = model(rt1, pt1)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            labels.append(y.cpu())
            pred.append(top_pred.cpu())

    labels = torch.cat(labels, dim=0)
    pred = torch.cat(pred, dim=0)

    return labels.numpy(), pred.numpy()


def plot_confusion_matrix_SR(cm, class_names):
    '''
    Plot the confusion matrix as a heatmap
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def print_report_SR(model, test_iterator, device, class_names=None):
    '''
    Print classification report and display confusion matrix
    '''
    labels, pred = predict_SR(model, test_iterator, device)
    
    cm = confusion_matrix(labels, pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(labels, pred))
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plot_confusion_matrix_SR(cm, class_names)





















