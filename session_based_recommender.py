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

                # Hard Positive Loss
                weights = torch.where((target == 1) & (outputs < 0.5), 10.0, 1.0)
                loss = (weights * criterion(outputs, target)).mean()

                # Hard Negative Loss
                # weights = torch.where(target == 0 & outputs > 0.5, 10.0, 1.0)
                # loss = (weights * criterion(outputs, target)).mean()

                # # Negative weighted Loss
                # weights = torch.where(target == 1, 1.0, 10.0)
                # loss = (weights * criterion(outputs, target)).mean()

                # Positive weighted Loss
                # weights = torch.where(target == 1, 10.0, 1.0)
                # loss = (weights * criterion(outputs, target)).mean()

                # 기본 Loss
                # loss = criterion(outputs, target)
                total_loss += loss.item()

                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {np.mean(loss_mean)}"
            )

            with torch.no_grad():
                top_k = 5
                recall_scores = []
                precision_scores = []
                for inputs, target in test_data:
                    outputs = self(inputs)

                    # Recall@k
                    for seq_idx, seq_row in enumerate(target):
                        target_indices = torch.nonzero(seq_row == 1).squeeze(-1)
                        target_items = set(target_indices.tolist())

                        _, top_k_indices = torch.topk(outputs[seq_idx], top_k)
                        predicted_items = set(top_k_indices.tolist())

                        recall = len(target_items & predicted_items) / len(target_items)
                        recall_scores.append(recall)

                        precision = len(target_items & predicted_items) / len(
                            predicted_items
                        )
                        precision_scores.append(precision)

                mean_recall_at_5 = sum(recall_scores) / len(recall_scores)
                print(f"Test Recall@{top_k}: {mean_recall_at_5:.4f}")
                mean_precision_at_5 = sum(precision_scores) / len(precision_scores)
                print(f"Test Precision@{top_k}: {mean_precision_at_5:.4f}")
