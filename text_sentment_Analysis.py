import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mysql.connector
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize global variables
df = None
clf = None
vectorizer = None
spark = None


# Connect to MySQL Database
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="customer_reviews"
    )


# Tkinter App Class
class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis App")
        self.geometry("800x600")
        self.create_widgets()

    def create_widgets(self):
        # Title
        title = tk.Label(self, text="Sentiment Analysis Application", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # File Upload Section
        upload_frame = tk.Frame(self)
        upload_frame.pack(pady=10)

        upload_label = tk.Label(upload_frame, text="Upload Dataset (CSV):")
        upload_label.pack(side=tk.LEFT)

        upload_button = tk.Button(upload_frame, text="Browse", command=self.load_dataset)
        upload_button.pack(side=tk.LEFT, padx=5)

        # Buttons for EDA and Model Training
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        eda_button = tk.Button(button_frame, text="Perform EDA", command=self.perform_eda)
        eda_button.pack(side=tk.LEFT, padx=10)

        train_button = tk.Button(button_frame, text="Train Model", command=self.train_model)
        train_button.pack(side=tk.LEFT, padx=10)

        # User Input for Sentiment Prediction
        input_frame = tk.Frame(self)
        input_frame.pack(pady=20)

        input_label = tk.Label(input_frame, text="Enter a review for sentiment prediction:")
        input_label.pack(side=tk.LEFT)

        self.review_entry = tk.Entry(input_frame, width=50)
        self.review_entry.pack(side=tk.LEFT, padx=10)

        predict_button = tk.Button(input_frame, text="Predict Sentiment", command=self.predict_sentiment)
        predict_button.pack(side=tk.LEFT, padx=10)

        # Display Section
        self.output_frame = tk.Frame(self)
        self.output_frame.pack(pady=20)

    def load_dataset(self):
        global df
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")

    def perform_eda(self):
        if df is not None:
            # Display dataset info
            eda_info = f"Dataset Info:\n{df.info()}\n\nMissing Values:\n{df.isnull().sum()}"
            messagebox.showinfo("EDA Results", eda_info)

            # Plot sentiment distribution
            df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
            plt.title("Sentiment Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.show()
        else:
            messagebox.showerror("Error", "Please upload a dataset first!")

    def train_model(self):
        global clf, vectorizer
        if df is not None:
            # Preprocess text and train model
            df['cleaned_review'] = df['review_text'].str.lower()

            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df['cleaned_review'])
            y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Encode sentiments as binary labels

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)

            # Evaluate Model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            messagebox.showinfo("Model Trained", f"Model trained successfully!\nAccuracy: {accuracy * 100:.2f}%")
        else:
            messagebox.showerror("Error", "Please upload a dataset first!")

    def predict_sentiment(self):
        global clf, vectorizer
        if clf is not None and vectorizer is not None:
            review_text = self.review_entry.get()
            if review_text:
                # Predict sentiment
                prediction = clf.predict(vectorizer.transform([review_text]))[0]
                sentiment = "Positive" if prediction == 1 else "Negative"

                # Save to MySQL Database
                conn = connect_to_db()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO reviews (review_text, sentiment) VALUES (%s, %s)",
                    (review_text, sentiment)
                )
                conn.commit()
                conn.close()

                messagebox.showinfo("Prediction Result", f"Sentiment: {sentiment}\nResult saved to database!")
            else:
                messagebox.showerror("Error", "Please enter a review for prediction!")
        else:
            messagebox.showerror("Error", "Please train the model first!")


# Initialize PySpark Session
def start_spark_session():
    global spark
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()


# Main Execution
if __name__ == "__main__":
    # Start PySpark Session
    start_spark_session()

    # Run Tkinter App
    app = SentimentApp()
    app.mainloop()
