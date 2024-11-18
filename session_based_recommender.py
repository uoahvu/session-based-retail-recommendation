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

    def forward(self, input, mask):
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

        last_true_indices = torch.sum(mask, 1) - 1
        batch_indices = torch.arange(output.size(0))
        output = output[batch_indices, last_true_indices]

        output = self.out_layer(output)
        output = self.sigmoid(output)
        return output

    def train_model(self, train_data, test_data, opt, candidate_dict):
        criterion_crossentropy = nn.CrossEntropyLoss()
        criterion_ranking = nn.MarginRankingLoss(margin=0.3)

        total_loss = 0
        for epoch in range(self.num_epochs):
            loss_mean = []
            for i, (inputs, target, mask) in enumerate(train_data):
                ranking_loss = 0
                opt.zero_grad()
                outputs = self(inputs, mask)

                # MarginRankingLoss
                # y = 1
                # for seq_idx, seq_row in enumerate(target):
                #     positive_x = outputs[seq_idx][seq_row == 1.0]
                #     negetive_seq = outputs[seq_idx][seq_row == 0.0]
                #     negetive_mean = torch.mean(negetive_seq)
                #     negetive_x = negetive_mean.expand(positive_x.size(0))
                #     target_y = torch.ones((negetive_x.size(0),))

                #     ranking_loss += criterion_ranking(positive_x, negetive_x, target_y)

                # Hard Positive Loss
                # weights = torch.where((target == 1) & (outputs < 0.5), 10.0, 1.0)
                # loss = (weights * criterion(outputs, target)).mean()

                # Hard Negative Loss
                # weights = torch.where(target == 0 & outputs > 0.5, 10.0, 1.0)
                # loss = (weights * criterion(outputs, target)).mean()

                # Negative weighted Loss
                # weights = torch.where(target == 1, 1.0, 10.0)
                # weighted_loss = (weights * criterion(outputs, target)).mean()

                # Positive weighted Loss
                weights = torch.where(target == 1, 10.0, 0.0)
                weighted_loss = torch.mean(
                    (weights * criterion_crossentropy(outputs, target))
                )

                # 기본 Loss
                # loss = criterion(outputs, target)

                loss = ranking_loss + weighted_loss
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
                for inputs, target, mask in test_data:
                    outputs = self(inputs, mask)

                    for seq_idx, seq_row in enumerate(target):
                        # Candidate Generation
                        candidate_pcategory = (
                            torch.unique(
                                inputs[seq_idx][: torch.sum(mask[seq_idx]), -1]
                            )
                            .to(torch.int64)
                            .tolist()
                        )
                        candidate_items = torch.unique(
                            torch.tensor(
                                [
                                    value
                                    for key in candidate_pcategory
                                    if key in candidate_dict
                                    for value in candidate_dict[key]
                                ]
                            )
                        )
                        _, top_k_relative_indices = torch.topk(
                            outputs[seq_idx][candidate_items], top_k
                        )
                        top_k_indices = candidate_items[top_k_relative_indices]

                        predicted_items = set(top_k_indices.tolist())

                        target_indices = torch.nonzero(seq_row == 1).squeeze(-1)
                        target_items = set(target_indices.tolist())

                        # Recall@k
                        recall = len(target_items & predicted_items) / len(target_items)
                        recall_scores.append(recall)

                        # precision@k
                        precision = len(target_items & predicted_items) / len(
                            predicted_items
                        )
                        precision_scores.append(precision)

                mean_recall_at_5 = sum(recall_scores) / len(recall_scores)
                print(f"Test Recall@{top_k}: {mean_recall_at_5:.4f}")
                mean_precision_at_5 = sum(precision_scores) / len(precision_scores)
                print(f"Test Precision@{top_k}: {mean_precision_at_5:.4f}")
