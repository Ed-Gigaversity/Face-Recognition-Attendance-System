import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Step 1: Load the dataset with the 'crop' column
df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv")

# Step 2: Fit the LabelEncoder on the crop names
le = LabelEncoder()
le.fit(df['crop'])

# ✅ Step 3: Save the label encoder to a file
dump(le, 'label_encoder.pkl')

print("✅ label_encoder.pkl has been created successfully.")
