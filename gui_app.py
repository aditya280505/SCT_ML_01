# gui_app.py
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model/house_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_price():
    try:
        sqft = float(entry_sqft.get())
        bed = int(entry_bed.get())
        bath = int(entry_bath.get())

        data = np.array([[sqft, bed, bath]])
        data_scaled = scaler.transform(data)
        price = model.predict(data_scaled)[0]

        result_label.config(text=f"Predicted Price: ₹ {int(price):,}")
    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# GUI setup
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("350x300")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Square Footage:").pack(pady=5)
entry_sqft = tk.Entry(root)
entry_sqft.pack()

tk.Label(root, text="Bedrooms:").pack(pady=5)
entry_bed = tk.Entry(root)
entry_bed.pack()

tk.Label(root, text="Bathrooms:").pack(pady=5)
entry_bath = tk.Entry(root)
entry_bath.pack()

tk.Button(root, text="Predict Price", command=predict_price, bg="#4CAF50", fg="white").pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="#f0f0f0")
result_label.pack()

root.mainloop()

# python gui_app.py             (run this in terminal)