import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
df = pd.read_csv("train.csv")
df.drop("Employee ID", axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict function
def get_user_input_and_predict():
    user_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            print(f"\nSelect {col} from: {list(options)}")
            while True:
                val = input(f"{col}: ").strip()
                if val in options:
                    user_data[col] = val
                    break
                else:
                    print("Invalid input. Try again.")
        else:
            user_data[col] = float(input(f"Enter {col} (numeric): "))

    # Create a DataFrame
    input_df = pd.DataFrame([user_data])

    # Encode the input
    for col in label_encoders:
        if col == "Attrition":
            continue
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    pred = model.predict(input_df)[0]
    result = label_encoders["Attrition"].inverse_transform([pred])[0]
    print(f"\n✅ Prediction: This employee will likely '{result}'.")

# Run the prediction
get_user_input_and_predict()
