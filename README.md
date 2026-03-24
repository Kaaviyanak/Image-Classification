# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

## Dataset
The FashionMNIST dataset consists of 70,000 grayscale images of size 28 × 28 pixels.

The dataset has 10 classes, representing different clothing categories such as T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

Training set: 60,000 images.

Test set: 10,000 images.

Images are preprocessed (normalized and converted to tensors) before being passed into the CNN.

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/9dd756d3-a1f3-48c3-b2bc-30ac4717949a" />

## DESIGN STEPS

### STEP 1:
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name:KAAVIYAN K
### Register Number:212224240066
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
```
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model

def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: KAAVIYAN K')
        print('Register Number: 212224240066')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

Include the Training Loss per epoch
<img width="473" height="283" alt="image" src="https://github.com/user-attachments/assets/25dc3c57-ee86-4d0c-ad69-dfc8807cafaa" />


### Confusion Matrix

Include confusion matrix here
<img width="818" height="776" alt="image" src="https://github.com/user-attachments/assets/863f1b28-542c-4c5b-b052-80d0b1a84980" />


### Classification Report

Include Classification Report here
<img width="589" height="418" alt="image" src="https://github.com/user-attachments/assets/146ba867-bd05-418a-ae54-6c506d9baf5f" />



### New Sample Data Prediction

Include your sample input and output 
<img width="590" height="619" alt="image" src="https://github.com/user-attachments/assets/3f134e5b-bcb4-4910-ba00-be701dbf3908" />


## RESULT

Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
