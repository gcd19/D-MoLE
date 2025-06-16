import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from internvl.model.autoencoder import AutoEncoder


def train_autoencoder(
    model,
    train_embeddings,
    num_epochs=20,
    batch_size=1024,
    patience=10,  # Early stopping patience
):
    dataset = TensorDataset(train_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Define the cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-3
    )

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data in dataloader:
            inputs = data[0].cuda()  # Ensure input is on GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        total_loss /= len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.8f}")

        # Step the scheduler after each epoch
        scheduler.step()

        # Check for early stopping condition
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
            break

    return model


# L2 normalization function
def l2_normalize(embeddings):
    norm = embeddings.norm(p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norm
    return normalized_embeddings


# try to load the autoencoder model from the disk, if not exist, train and save the autoencoder
def load_or_train_autoencoder(
    folder,
    embedding,
    input_dim,
    hidden_dim,
    model_path,
    num_epochs=20,
):
    autoencoder = AutoEncoder(input_dim, hidden_dim).cuda()

    if os.path.exists(model_path):
        print(f"Loading pre-trained autoencoder for {folder}...")
        autoencoder.load_state_dict(torch.load(model_path))
    else:
        print(f"Training new autoencoder for {folder}...")
        autoencoder = train_autoencoder(
            autoencoder,
            embedding,
            num_epochs,
        )
        torch.save(autoencoder.state_dict(), model_path)
        print(f"Autoencoder for {folder} saved at {model_path}")

    return autoencoder


def create_reconstruction_loss_quantile_table(embeddings, autoencoders, folders):
    """create the reconstruction loss quantile table"""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    # store the results
    results = {}

    for folder in folders:
        print(f"Computing reconstruction losses for {folder}...")

        # compute the reconstruction loss
        current_losses = autoencoders[folder].compute_reconstruction_loss(
            embeddings[folder]
        )

        # compute the quantiles
        quantile_values = torch.quantile(current_losses, torch.tensor(quantiles).cuda())

        # store the results
        results[folder] = {
            f"Q{int(q*100)}": val.item() for q, val in zip(quantiles, quantile_values)
        }

        results[folder]["mean"] = current_losses.mean().item()
        results[folder]["std"] = current_losses.std().item()
        results[folder]["min"] = current_losses.min().item()
        results[folder]["max"] = current_losses.max().item()

    # convert to DataFrame
    df = pd.DataFrame(results).T

    # rearrange the columns
    columns_order = [
        "min",
        "Q10",
        "Q25",
        "Q50",
        "Q75",
        "Q90",
        "Q95",
        "Q99",
        "max",
        "mean",
        "std",
    ]
    df = df[columns_order]

    return df


# Base directory paths for embeddings
embeddings_dir = f"embeddings"

folders = [
    "vizwiz_caption",
    "skvg",
    "textcaps",
    "iconqa",
    "ocrvqa",
    "flickr30k",
    "vizwiz",
    "kvqa",
    "pmcvqa",
]

# Load embedding data for each task
embeddings = {
    folder: l2_normalize(
        torch.load(f"{embeddings_dir}/{folder}/embeddings.pt").to(torch.float32).cuda()
    )
    for folder in folders
}

# train and save the autoencoder for each task, or load the existing autoencoder
hidden_dim = (
    128  # the dimension of the hidden layer can be adjusted according to the need
)
autoencoders = {}

for folder, embedding in embeddings.items():
    input_dim = embedding.shape[1]
    model_path = f"autoencoder_models/{folder}/autoencoder.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # load or train the autoencoder
    autoencoder = load_or_train_autoencoder(
        folder,
        embedding,
        input_dim,
        hidden_dim,
        model_path,
        num_epochs=1000,
    )

    autoencoders[folder] = autoencoder

# create the reconstruction loss quantile table
print("\nCreating reconstruction loss quantile table...")
quantile_table = create_reconstruction_loss_quantile_table(
    embeddings, autoencoders, folders
)

# save the quantile table
output_path = f"autoencoder_models/reconstruction_loss_quantiles.csv"
quantile_table.to_csv(output_path)
print(f"Reconstruction loss quantile table saved to: {output_path}")

# print the quantile table
print("\nReconstruction Loss Quantile Table:")
print(quantile_table)
