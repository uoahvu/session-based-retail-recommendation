import torch
import torch.nn as nn
import numpy as np


class SessionLSTM(nn.Module):
    def __init__(
        self,
        num_epochs,
        num_items,
        num_categorys,
        num_pcategorys,
        hidden_size,
        num_layers=1,
    ):
        super(SessionLSTM, self).__init__()
        self.num_epochs = num_epochs
        self.num_items = num_items
        self.num_category = num_categorys
        self.num_parent_category = num_pcategorys

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_item = nn.Embedding(num_items, hidden_size)
        self.embedding_category = nn.Embedding(num_categorys, hidden_size)
        self.embedding_parent_category = nn.Embedding(num_pcategorys, hidden_size)
        self.lstm_layer = nn.LSTM(
            hidden_size * 3, hidden_size, num_layers, batch_first=True
        )
        self.out_layer = nn.Linear(hidden_size, num_items)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        embedded_item = self.embedding_item(input[:, :, 0].long())
        embedded_category = self.embedding_category(input[:, :, 1].long())
        embedding_pcategory = self.embedding_parent_category(input[:, :, 2].long())
        embedded = torch.cat(
            (
                torch.cat((embedded_item, embedded_category), -1),
                embedding_pcategory,
            ),
            -1,
        )
        output, _ = self.lstm_layer(embedded)
        output = self.out_layer(output[:, -1, :])
        output = self.sigmoid(output)
        return output

    def train_model(self, train_data, test_data, opt):
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        for epoch in range(self.num_epochs):
            loss_mean = []
            for i, (inputs, target) in enumerate(train_data):
                opt.zero_grad()
                outputs = self(inputs)

                loss = criterion(outputs, target)
                total_loss += loss.item()

                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

                # print(
                #     f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_data)}], {loss}"
                # )
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {np.mean(loss_mean)}"
            )

            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, target in test_data:
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                print(f"Test ACC: {correct / total}")
