import pandas as pd
from pandasgui import show
import tkinter as tk
import pickle
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox
from tkinter import filedialog

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
data.columns = ['class','alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline']
bWidth = 30
bHeight = 2

#Split data into 2 sets (train, test)
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

#Building model
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy} ") #string formatting

#Load model from file
with open('model.pkl', 'rb') as file: #read binary
    loaded_model = pickle.load(file)

#Saving model to five
with open('model.pkl', 'wb') as file: #write
    pickle.dump(model, file)

def train_model():
    global model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    messagebox.showinfo("Training", "Model trained")

def test_model():
    accuracy = model.score(X_test, y_test)
    messagebox.showinfo("Testing", f"Model accuracy: {accuracy}")

# New data from csv
def predict_new_data():
    global new_data
    file_path = filedialog.askopenfilename(title="Select Data File")
    new_data = pd.read_csv(file_path)

    prediction = model.predict(new_data)
    messagebox.showinfo("Prediction", f"Predicted class: {prediction}")

# Browse_data
def browse_data():
    show(data)

# Wykres
def wykres():
    plt.plot(data['alcohol'],data['color_intensity'], 'bo')
    plt.ylabel('Color Intensity')
    plt.xlabel('Alcohol')
    plt.show()

#GUI
root = tk.Tk()
root.title("Wine App s24341")
root.geometry("800x600")
root.config(bg="#DBE8F2")

text_label = tk.Label(root, text="Wine APP - s24341", font=("Arial", 36), fg="#5B77C2", bg="#DBE8F2")
text_label.pack()

train_button = tk.Button(root, text="Train", command=train_model, width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
train_button.pack()

rebuild_button = tk.Button(root, text="Rebuild", command=train_model, width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
rebuild_button.pack()

test_button = tk.Button(root, text="Test", command=test_model, width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
test_button.pack()

predict_button = tk.Button(root, text="Predict New Data", command=predict_new_data, width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
predict_button.pack()

plot_button = tk.Button(root, text="Plot Data", command=wykres, width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
plot_button.pack()

browse_button = tk.Button(root, text="Browse Data", command=browse_data, width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
browse_button.pack()

read_button = tk.Button(root, text="Read from Database", width=bWidth, height=bHeight,fg="#DBE8F2", bg="#5B77C2")
read_button.pack()


root.mainloop()