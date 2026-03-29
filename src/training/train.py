import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.globals.globals import BASE_DIR
from src.models.model import ModelManager

# ===== Accessing Data =====
X = pd.read_csv(BASE_DIR / "data/raw/inputs.csv", header=None).astype(float)
y = pd.read_csv(BASE_DIR / "data/raw/targets.csv", header=None).astype(float)
y = np.argmax(y, axis=1)

# ===== Preparing Data =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# ===== Model =====
model = ModelManager()
model = model.build()

# ===== Training =====
model.fit(x=X_train, y=y_train, batch_size=16, epochs=50)

# ===== Validation =====
print("================== VALIDATION ==================")
predicts = model.predict(X_test)
pred_classes = np.argmax(predicts, axis=1)
accuracy = accuracy_score(y_test, pred_classes)
print(f"accuracy: {accuracy*100:.2f}")

# ===== Save =====
save_model = str(input("Do you want to save the model? (Y/n): "))
if save_model.upper() == "Y":
    model.save(BASE_DIR / "models/model.keras")

# Iris Setosa = [1, 0, 0]
# Iris Versicolour = [0, 1, 0]
# Iris Virginica = [0, 0, 1]