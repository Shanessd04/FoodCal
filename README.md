# 🥗 FoodCal: Food Image to Nutrition Breakdown

[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Deployed%20with-Streamlit-ff4b4b?style=flat&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)

---

## 📌 Project Overview

**FoodCal** is a deep learning-powered nutrition analysis app that takes an image of Indian food and predicts:
- 🍛 Dish Name
- 🧬 Nutrition Breakdown: **Calories, Protein, Fat, Carbohydrates**

Built using **TensorFlow (MobileNetV2)** and deployed via a sleek **Streamlit web app**, FoodCal offers instant nutrition insights from just a photo.

---

## 🧠 Model Highlights

- ✅ **Model:** MobileNetV2 with fine-tuning
- ✅ **Input Shape:** 224×224×3 (RGB food images)
- ✅ **Output:** Softmax predictions across Indian food classes
- ✅ **Dataset:** Custom Indian food dataset (~20 classes, images only)
- ✅ **Nutrition Mapping:** Linked with a manually curated CSV file

---

## ⚙️ How it Works

1. **User uploads a food image**
2. The model predicts the dish (e.g., *Aloo Gobi*)
3. The system looks up nutrition from a CSV file (`indian_food_nutrition.csv`)
4. Output:
   - Dish name + confidence
   - Nutrition per serving (calories, protein, fat, carbs)
