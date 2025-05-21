# src/predict.py

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.preprocessing import image

# === Paths ===
MODEL_PATH = "models/food_model.keras"
CLASS_INDEX_PATH = "models/class_indices.json"
NUTRITION_CSV = "data/nutrition.csv"

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Load Class Indices ===
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# === Reverse Class Mapping ===
index_to_class = {v: k for k, v in class_indices.items()}

# === Load Nutrition Data ===
nutrition_df = pd.read_csv(NUTRITION_CSV)

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def predict_food(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = index_to_class[predicted_index]
    confidence = predictions[0][predicted_index]

    print(f"\n🍛 Predicted Dish: {predicted_class} ({confidence:.2f})")

    # Get nutrition info
    match = nutrition_df[nutrition_df['food_name'].str.lower() == predicted_class.lower()]
    if not match.empty:
        row = match.iloc[0]
        print(f"📊 Nutrition Breakdown per serving of {predicted_class.title()}:")
        print(f"   🥄 Calories: {row['calories']} kcal")
        print(f"   🧬 Protein : {row['protein']} g")
        print(f"   🧈 Fat     : {row['fat']} g")
        print(f"   🍞 Carbs   : {row['carbohydrates']} g")
    else:
        print("⚠️ Nutrition information not found for this dish.")

# === Run Prediction ===
if __name__ == "__main__":
    test_image = "/Users/shanes/FoodCal/data/gulab-jamun-recipe-146.jpg"  # Replace with your image path
    predict_food(test_image)
