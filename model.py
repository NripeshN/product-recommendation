import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Choose the device based on availability of GPU or MPS (Apple's Metal Performance Shaders)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class RetailDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.users = self.data['CustomerID'].unique()
        self.items = self.data['StockCode'].unique()
        self.user_to_index = {user: idx for idx, user in enumerate(self.users)}
        self.item_to_index = {item: idx for idx, item in enumerate(self.items)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_idx = self.user_to_index[row['CustomerID']]
        item_idx = self.item_to_index[row['StockCode']]
        return torch.tensor(user_idx), torch.tensor(item_idx)

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # Adjust the input feature size of the linear layer to 2 * emb_size
        self.fc = nn.Linear(2 * emb_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_embedding = self.user_emb(user)  # Shape: [batch_size, emb_size]
        item_embedding = self.item_emb(item)  # Shape: [batch_size, emb_size]
        # Concatenate user and item embeddings
        combined = torch.cat((user_embedding, item_embedding), dim=1)  # Shape: [batch_size, 2 * emb_size]
        output = self.fc(combined)  # Pass through the linear layer
        return self.sigmoid(output)  # Return the sigmoid output

def train_model(model, epochs, data_loader):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for user, item in data_loader:
            user, item = user.to(device), item.to(device)
            optimizer.zero_grad()
            prediction = model(user, item)
            label = torch.ones_like(prediction, device=device)  # Assuming these are positive samples
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Total Loss: {total_loss / len(data_loader)}')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

if __name__ == "__main__":
    dataset = RetailDataset('cleaned_data.csv')
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    model = RecommendationModel(len(dataset.users), len(dataset.items))
    train_model(model, 10, data_loader)
    save_model(model, 'recommendation_model.pth')