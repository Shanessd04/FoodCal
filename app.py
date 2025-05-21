import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# Load model and nutrition CSV
model = tf.keras.models.load_model("models/food_model.keras")
nutrition_df = pd.read_csv("/Users/shanes/FoodCal/data/nutrition.csv")

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict_dish(image):
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]
    predicted_index = np.argmax(predictions)
    class_names = nutrition_df['food_name'].unique()
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index]
    return predicted_class, confidence

# UI
st.title("🍽️ FoodCal: Nutrition Estimator from Food Image")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        food_name, confidence = predict_dish(image)

    st.success(f"🍛 Predicted Dish: **{food_name}** ({confidence:.2f} confidence)")

    # Show nutrition
    row = nutrition_df[nutrition_df['food_name'].str.lower() == food_name.lower()]
    if not row.empty:
        cal = row.iloc[0]["calories"]
        prot = row.iloc[0]["protein"]
        fat = row.iloc[0]["fat"]
        carbs = row.iloc[0]["carbohydrates"]

        st.markdown("### 📊 Nutrition Breakdown (per serving):")
        st.write(f"🥄 **Calories:** {cal} kcal")
        st.write(f"🧬 **Protein:** {prot} g")
        st.write(f"🧈 **Fat:** {fat} g")
        st.write(f"🍞 **Carbs:** {carbs} g")
    else:
        st.warning("Nutrition data not found.")
