# 🦠🩺 Pneumonia Detection Model using Transfer Learning 🔍

This project uses a transfer learning approach to predict pneumonia from chest X-ray images. By utilizing a pre-trained convolutional neural network, the model benefits from previously learned features, allowing it to effectively identify intricate patterns in medical imaging. 


## 💡 Motivation 
- Early and accurate detection of pneumonia can significantly improve patient outcomes.
- By leveraging advanced machine learning techniques, this project seeks to provide a reliable tool for healthcare professionals to diagnose pneumonia from chest X-rays efficiently.

## 📊 Dataset 
The dataset used in this project consists of chest X-ray images. The images are categorized into two classes:
- Normal
- Pneumonia

The dataset should be organized into the following directory structure:

![image](https://github.com/mayurd8862/Pneumonia-Detection-using-Transfer-Learning/assets/113239727/88677471-e9d1-4b5e-8429-6dedace897f1)

## 🏗️ Model Architecture 
The model uses the ResNet50V2 architecture, a powerful convolutional neural network pre-trained on the ImageNet dataset. Which contains over 14 million images and 1000 classes, making it highly effective for image classification tasks. The top layers of ResNet50V2 are replaced with custom layers to adapt it for the binary classification task of pneumonia detection.

## ⚙️ Installation 
To run this project locally, follow these steps:

1. 🛠️ Clone the repository:
    ```bash
    git clone https://github.com/mayurd8862/Pneumonia-Detection-using-Transfer-Learning.git
    ```
2. 📁 Navigate to the project directory:
    ```bash
    cd Pneumonia-Detection-using-Transfer-Learning
    ```
3. 💻 Create a virtual environment and activate it:
    ```bash
    python -m venv env
    env\Scripts\activate
    ```
4. 📦 Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage 
1. 📂 Prepare the dataset and place it in the `data` folder as described above.
2. 🏃 Run the training script to train the model:
    ```bash
    python train.py
    ```
3. 🖼️ To predict pneumonia from a new chest X-ray image, use the prediction script:
    ```bash
    Streamlit run app.py
    ```

## 📈 Results 
The model's performance is evaluated using accuracy, precision, recall, and F1-score. Detailed results and model evaluation metrics will be displayed upon training completion.

![image](https://github.com/mayurd8862/Pneumonia-Detection-using-Deep-Learning/assets/113239727/c0a38f80-f77b-4e29-b655-dcf68a4b28d1)

## 🤝 Contributing 
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## 📜 License 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


