# Moody

This repository contains the source code for Moody, a real-time emotion detection model integrated into a web application. <br/>
The project is built, mainly using Python and Flask, providing a user-friendly interface.

## 📸 Demo

![App Screenshot](https://github.com/Soukaina235/moody-real-time-emotion-detection-web-app/blob/main demo/welcome.png)
![App Screenshot](https://github.com/Soukaina235/moody-real-time-emotion-detection-web-app/blob/main/demo/detection1.png)
![App Screenshot](https://github.com/Soukaina235/moody-real-time-emotion-detection-web-app/blob/main/demo/detection2.png)
![App Screenshot](https://github.com/Soukaina235/moody-real-time-emotion-detection-web-app/blob/main/demo/about1.png)
![App Screenshot](https://github.com/Soukaina235/moody-real-time-emotion-detection-web-app/blob/main/demo/about2.png)

## 🛠️ Installation

1. **Tools Used:**<br />
   Python for building the model<br />
   HTML, CSS, JavaScript, and Bootstrap for the frontend<br />
   Flask for deploying the model<br />

2. **Requirements:**

```
opencv
numpy
kera
csv
pandas
PIL
tqdm
matplotlib
os
visualkeras
```

## ⭐Features

- Real-time Emotion Detection: The project incorporates a real-time emotion detection model.
- Face Detection: The model utilizes the Haar cascade model to detect faces in real-time camera feeds.
- Emotion Display: Detected emotions are displayed in a square overlay on top of the detected face.
- Emotion Saving: Predictions are saved into a CSV file for further analysis and tracking.

## 🌟 Datasets & Model Architecture

Our project utilized two different datasets: [FER 2013 from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and [CKextended]().<br>
We experimented with two different model architectures on each dataset, resulting in four different models in total. <br>
After thorough evaluation, we selected the best-performing model for integration into our web application.

## 🤝 Members

[Hamdoune Chaymae](https://github.com/Hchaymae)<br />
[fzhachchane](https://github.com/fzhachchane)
