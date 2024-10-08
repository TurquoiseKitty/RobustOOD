'''Implementation of Example 1-2'''

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



# Define the model
class RegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.randn(1,input_size))
        self.sigma = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return torch.exp(self.sigma + torch.sum(self.theta * x, dim=1))

# Define the loss function
def loss_fn(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def train_q(x_data,r_data,x2_data,x3_data):
    dim_r = r_data.shape[1]
    input_size = x_data.shape[1]
    q2_UQ = np.zeros((x_data.shape[0],dim_r))
    q2_fit = np.zeros((x2_data.shape[0],dim_r))
    q2_test = np.zeros((x3_data.shape[0],dim_r))
    for j in range(dim_r):
    
        # Create the model
        model = RegressionModel(input_size)
        # Define the optimizer
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Convert the data to PyTorch tensors
        x = torch.tensor(x_data, dtype=torch.float32)
        r = torch.tensor(r_data[:,j]**2, dtype=torch.float32)

        # Create a PyTorch dataset and dataloader
        dataset = TensorDataset(x, r)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            for batch_x, batch_r in dataloader:
                # Forward pass
                y_pred = model(batch_x)

                # Compute the loss
                loss = loss_fn(y_pred, batch_r)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print the loss every 10 epochs
            if epoch % 100 == 99:
                print(f"Epoch {epoch}, loss = {loss.item()}")

        # Get the learned parameters


        model.eval()
        with torch.no_grad():
            q2_UQ[:,j] = model(torch.tensor(x_data, dtype=torch.float32)).numpy()
            q2_fit[:,j] = model(torch.tensor(x2_data, dtype=torch.float32)).numpy()
            q2_test[:,j] = model(torch.tensor(x3_data, dtype=torch.float32)).numpy()

    return q2_fit, q2_UQ, q2_test
