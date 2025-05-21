import os
import pandas as pd

# Set path to your dataset directory
dataset_path = "/Users/shanes/FoodCal/new_dataset"  # change this if your folder is named differently

# Get class folder names
classes = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

# Create a basic DataFrame template
nutrition_data = pd.DataFrame({
    "food_name": classes,
    "calories": [0] * len(classes),
    "protein": [0] * len(classes),
    "fat": [0] * len(classes),
    "carbohydrates": [0] * len(classes),
})

# Save CSV
output_path = "data/indian_food_nutrition.csv"
nutrition_data.to_csv(output_path, index=False)
print(f"✅ Nutrition CSV created at: {output_path}")
