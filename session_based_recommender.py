import torch
import torch.nn as nn


class SessionLSTM(nn.Module):
    def __init__(self, num_items, hidden_size, output_size, num_layers=1):
        super(SessionLSTM, self).__init__()
        self.num_items = num_items
        self.num_category = 0
        self.num_parent_category = 0

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_item = nn.Embedding(num_items, hidden_size)
        self.embedding_category = nn.Embedding(num_category, hidden_size)
        self.embedding_parent_category = nn.Embedding(num_parent_category, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded_item = self.embedding_item(input[:, :, 0])
        embedded_category = self.embedding_type(input[:, :, 1])
        embedding_parent_category = self.embedding_type(input[:, :, 2])
        embedded = torch.cat(
            (
                torch.cat((embedded_item, embedded_category), -1),
                embedding_parent_category,
            ),
            -1,
        )
        output, _ = self.lstm(embedded)  # torch.Size([31, 5, 128])
        output = self.out(output[:, -1, :])  # 세션의 마지막 예측만 가져옴
        return output

    def train(self, train_data, test_data):
        print("n_items:", self.num_items)
        model = self(self.num_items, 128, self.num_items)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 1000
        total_loss = 0
        for epoch in range(num_epochs):
            start_time = time.time()
            for i, (inputs, target) in enumerate(train_data):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader_sample)}], {loss}"
                )

        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, target in test_data:
                print(inputs)
                outputs = model(inputs)
                print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                print(predicted)
                print(target)
                correct += (predicted == target).sum().item()

            print(f"Accuracy of the model on the test data: {100 * correct / total}%")
