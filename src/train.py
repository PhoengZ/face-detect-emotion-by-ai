import torch
from data_loader import get_loader
from model import EmotionCNN
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

model = EmotionCNN(num_class=7).to(device)

criteria = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader  = get_loader(batch_size=64)

num_of_training = 100

best_acc = 0

for epoch in range(num_of_training):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader)
    for batch in loop:
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f'Epoch: {epoch+1}/{num_of_training}')
        loop.set_postfix(loss=loss.item())
    
    model.eval()
    correct = total = 0
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
            loss = criteria(outputs, labels)
            test_loss += loss.item()
            max_value, index = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (index == labels).sum().item()
    acc = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Accuracy on test set: {acc:.4f}%, Average test loss: {avg_test_loss:.4f}')

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Save successfully")
print(f"Training completed with accuracy: {best_acc:.4f}%")