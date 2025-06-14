
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

train_file = np.load('/kaggle/input/cse-251-b-2025/train.npz')
train_data = train_file['data']
print("train_data's shape", train_data.shape)

test_file = np.load('/kaggle/input/cse-251-b-2025/test_input.npz')
test_data = test_file['data']
print("test_data's shape", test_data.shape)

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx] #return the scene
        # Getting 50 historical timestamps and 60 future timestamps
        hist = scene[:, :50, :].copy()    # (agents=50, time_seq=50, 6) #for all agents, get the first 50 time steps and all features
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)  # (60, 2) #get the next 60 time steps and 2 position features of ego

        # Data augmentation(only for training)
        if self.augment:
            if np.random.rand() < 0.5: #rotate half of the time
                theta = np.random.uniform(-np.pi, np.pi) #rotate the scene a random amount
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                # Rotate the historical trajectory and future trajectory
                hist[..., :2] = hist[..., :2] @ R # matrix multiplication
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5: # mirror half of the time
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        # Use the last timeframe of the historical trajectory as the origin, shift
        origin = hist[0, 49, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin
        future = future - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale

        heading = hist[..., 4]
        heading_sin = torch.sin(torch.tensor(heading))
        heading_cos = torch.cos(torch.tensor(heading))
        heading_feat = torch.stack([heading_sin, heading_cos], dim=-1)
        hist = torch.cat([torch.tensor(hist), heading_feat], dim=-1)

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32), #create a tensor of history data
            y=future.type(torch.float32), #check if future is the right data type
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0), #create a tensor with origin position shifted
            scale=torch.tensor(self.scale, dtype=torch.float32), #scale if needed
            edge_index=edge_index
        )

        return data_item


class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0):
        """
        data: Shape (N, 50, 50, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        """
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Testing data only contains historical trajectory
        scene = self.data[idx]  # (50, 50, 6)
        hist = scene.copy()

        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        heading = hist[..., 4]
        heading_sin = torch.sin(torch.tensor(heading))
        heading_cos = torch.cos(torch.tensor(heading))
        heading_feat = torch.stack([heading_sin, heading_cos], dim=-1)
        hist = torch.cat([torch.tensor(hist), heading_feat], dim=-1)

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
            edge_index=edge_index
        )
        return data_item

torch.manual_seed(251)
np.random.seed(42)

scale = 7.0

N = len(train_data)
val_size = int(0.1 * N)
train_size = N - val_size

train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=False)
val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

# Set device for training speedup
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')

torch.backends.cudnn.benchmark = True

import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        # Create constant positional encoding matrix with shape (max_len, dim)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # (dim/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)  # buffer, not a parameter

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, dim)
        Returns positional embeddings broadcasted to x's seq_len.
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :].to(x.device)

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=2, timesteps=60):
        super().__init__()
        self.timesteps = timesteps
        self.input_dim = hidden_dim + 4  # +2 for last velocity

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)  # Predict delta x, delta y
        )

    def forward(self, z, last_vel, last_pos):
        """
        z:        (B, hidden_dim)
        last_vel: (B, 2)
        last_pos: (B, 2)

        Returns: (B, timesteps, 2) absolute positions
        """
        z_aug = torch.cat([z, last_pos, last_vel], dim=-1)  # (B, hidden_dim + 2)

        # Repeat over timesteps
        z_seq = z_aug.unsqueeze(1).repeat(1, self.timesteps, 1)  # (B, T, H+2)
        out, _ = self.lstm(z_seq)  # (B, T, H)
        delta = self.mlp(out)  # (B, T, 2)

        # Cumulative sum of deltas to get trajectory
        pred = torch.cumsum(delta, dim=1) + last_pos.unsqueeze(1)  # (B, T, 2)

        return pred

class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, hidden=None):
        x = F.relu(self.input_proj(x))  # (B, T, hidden_dim)
        output, hidden = self.lstm(x, hidden)
        return output, hidden

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 50, hidden_dim))  # positional embeddings for timesteps
        self.pos_embedding = SinusoidalPositionalEmbedding(hidden_dim, max_len=50)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: (B * A, T, F) tensor where
           B = batch size,
           A = number of agents,
           T = timesteps,
           F = features (like x,y, vx, vy)
        Returns: (B * A, hidden_dim)
        """
        B_A, T, F = x.shape
        x = self.input_proj(x)  # (B*A, T, hidden_dim)
        pos_emb = self.pos_embedding(x)[:, :T, :]  # (1, T, hidden_dim)
        x = x + pos_emb

        # shape: (T, B*A, hidden_dim)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)  # (T, B*A, hidden_dim)

        # last timestep output as the agent embedding
        x = x[-1, :, :]  # (B*A, hidden_dim)

        x = self.norm(x)
        return x

device = 'cuda'

# Initialize models
# encoder = LSTMEncoder().to(device)
encoder = TransformerEncoder().to(device)
decoder = LSTMDecoder().to(device)
# to_output = EmbedToOutput().to(device)

optimizers = [
    torch.optim.Adam(encoder.parameters(), lr=5e-4),
    torch.optim.Adam(decoder.parameters(), lr=5e-4),
    # torch.optim.Adam(gnn.parameters(), lr=5e-4)
]
loss_fn = nn.MSELoss()

def train_one_epoch(dataloader):
    encoder.train()
    decoder.train()
    total_loss = 0
    total_loss_raw = 0

    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)

        x = batch.x.view(-1, 50, 8)  # (B*A, 50, 6)

        latent = encoder(x)

        B = batch.num_graphs
        A = 50
        latent = latent.view(B,A,-1)
        ego_latent = latent[:,0,:]

        x = batch.x.view(B, A, 50, 8)  # (B, A, T, F)
        ego_hist = x[:, 0]
        last_pos = ego_hist[:, 49, :2]                 # (B, 2)
        last_vel = ego_hist[:, 49, 2:4]

        preds = decoder(ego_latent, last_vel, last_pos)


        target = batch.y.view(-1, 60, 2)  # (B, 60, 2)

        # Compute normalized & raw loss
        preds_raw = preds * batch.scale.view(-1, 1, 1) + batch.origin.view(-1, 1, 2)
        target_raw = target * batch.scale.view(-1, 1, 1) + batch.origin.view(-1, 1, 2)

        loss = F.mse_loss(preds, target)
        loss_raw = F.mse_loss(preds_raw, target_raw)

        # print(f"TRAIN Norm Loss: {loss.item()}, Raw Loss: {loss_raw.item()}")

        # Backprop
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

        total_loss += loss.item()
        total_loss_raw += loss_raw.item()

    print(f"TRAIN EPOCH---- Norm Loss: {total_loss / len(dataloader)}, Raw Loss: {total_loss_raw / len(dataloader)}")

    return total_loss / len(dataloader)

def validate(dataloader):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_loss_raw = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)

            target = batch.y.view(-1, 60, 2)       # (B, 60, 2)
            x = batch.x.view(-1, 50, 8)  # (B*A, 50, 6)

            latent = encoder(x)

            B = batch.num_graphs
            A = 50
            latent = latent.view(B,A,-1)
            ego_latent = latent[:,0,:]

            x = batch.x.view(B, A, 50, 8)  # (B, A, T, F)
            ego_hist = x[:, 0]
            last_pos = ego_hist[:, 49, :2]                 # (B, 2)
            last_vel = ego_hist[:, 49, 2:4]                # (B, 2)

            # Forward pass
            preds = decoder(ego_latent, last_vel, last_pos)

            # Denormalize
            preds_raw = preds * batch.scale.view(-1, 1, 1) + batch.origin.view(-1, 1, 2)
            target_raw = target * batch.scale.view(-1, 1, 1) + batch.origin.view(-1, 1, 2)

            # Compute loss (both normalized and raw for monitoring)
            loss_norm = F.mse_loss(preds, target)
            loss_raw = F.mse_loss(preds_raw, target_raw)

            # print(f"VAL Norm Loss: {loss_norm.item()}, Raw Loss: {loss_raw.item()}")

            total_loss += loss_norm.item()
            total_loss_raw += loss_raw.item()
    print(f"VAL EPOCH---- Norm Loss: {total_loss / len(dataloader)}, Raw Loss: {total_loss_raw / len(dataloader)}")

    return total_loss / len(dataloader)

num_epochs = 200
patience = 5
best_val_loss = float('inf')
best_train_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(train_dataloader)
    val_loss = validate(val_dataloader)
    print(f"Epoch {epoch+1} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if val_loss < best_val_loss and train_loss < best_train_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        counter = 0
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_encoder': optimizers[0].state_dict(),
            'optimizer_decoder': optimizers[1].state_dict(),
            'val_loss': val_loss
        }, f"/kaggle/working/best_{epoch}_{val_loss:.4f}_lstm_gnn_model_with_pos_vel_heading.pth")
        print("Saved new best model")
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s)")

    if counter >= patience:
        print("Early stopping triggered")
        break

checkpoint_path = '/kaggle/working/best_55_0.18287447444163263_lstm_gnn_model_with_pos_vel_heading.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

test_dataset = TrajectoryDatasetTest(test_data, scale=scale)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

encoder.eval()
decoder.eval()

all_preds = []


with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Test Predictions"):
        batch = batch.to(device)

        x = batch.x.view(-1, 50, 8)

        latent = encoder(x)

        B = batch.num_graphs
        A = 50
        latent = latent.view(B,A,-1)
        ego_latent = latent[:,0,:]

        x = batch.x.view(B, A, 50, 8)  # (B, A, T, F)
        ego_hist = x[:, 0]
        last_pos = ego_hist[:, 49, :2]
        last_vel = ego_hist[:, 49, 2:4]

        # Forward pass
        preds = decoder(ego_latent, last_vel, last_pos)

        # Denormalize
        preds_raw = preds * batch.scale.view(-1, 1, 1) + batch.origin.view(-1, 1, 2)

        all_preds.append(preds_raw.cpu())

# Concatenate all scenes
all_preds = torch.cat(all_preds, dim=0)  # (2100, 60, 2)
all_preds = all_preds.reshape(-1, 2)     # (126000, 2)

# Write to CSV
df = pd.DataFrame(all_preds.numpy(), columns=["x", "y"])
df.insert(0, "index", np.arange(len(df)))  # Required index column

df.to_csv("epoch_49_trajectory_predictions_transformer_lstm.csv", index=False)
print(" Saved predictions to trajectory_predictions.csv")

