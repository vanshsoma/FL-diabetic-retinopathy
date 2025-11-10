# 🤖 Federated Learning for Diabetic Retinopathy Screening

This project demonstrates a privacy-preserving deep learning framework to screen for Diabetic Retinopathy (DR) using Federated Learning (FL). It uses the Flower framework to train a MobileNetV2 model on partitioned data from multiple "hospitals" without the patient data ever leaving the local client.

![Dashboard Screenshot]<img width="1908" height="837" alt="Screenshot 2025-11-08 102132" src="https://github.com/user-attachments/assets/332736d9-03e8-46b2-a2d8-211da441bd90" />
 
*(Note: This image shows a final 10-round run. Your current setup runs for 6 rounds).*

## 1. 🎯 The Problem: The AI-Privacy Paradox

* **The Disease:** Diabetic Retinopathy (DR) is a leading cause of preventable blindness, affecting millions.
* **The Solution:** AI can screen for DR by analyzing retinal scans, but this requires a massive, diverse dataset.
* **The Paradox:** Patient medical data is highly sensitive and protected by laws (like HIPAA, GDPR, DPDPA). It is illegal and unethical to centralize this data on a single server for training.

## 2. 💡 The Solution: Federated Learning

Federated Learning (FL) solves this problem by **bringing the model to the data, not the data to the model.**



<img width="682" height="468" alt="image" src="https://github.com/user-attachments/assets/3535ef8a-1ca9-4d2c-ad73-de9309d8a3a9" />



1.  **Server:** A central server initializes a global model (e.g., MobileNetV2).
2.  **Clients (Hospitals):** Each hospital (client) downloads the global model.
3.  **Local Training:** The model is trained *locally* on the hospital's private patient data. The data **never** leaves the hospital.
4.  **Aggregation:** The clients send their updated model *weights* (not data) back to the server.
5.  **Improvement:** The server averages these weights to create a smarter, improved global model.
6.  **Repeat:** This process is repeated for multiple rounds.

## 3. 🛠️ Technology Stack

* **FL Framework:** [Flower (flwr)](https://flower.ai/)
* **ML Model:** [TensorFlow](https://www.tensorflow.org/) / Keras
* **Model Architecture:** MobileNetV2
* **Web Dashboard:** [Streamlit](https://streamlit.io/)
* **Data Handling:** Pandas, NumPy, Scikit-learn, OpenPyXL
* **Core:** Python 3.11

