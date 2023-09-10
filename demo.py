from tkinter import Tk, Button, Label, filedialog
import pandas as pd
import joblib
import re
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

def cleaning(text):
    text = re.sub(r"(?:\n|\r)", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я ]+", "", text).strip()
    text = text.lower()
    return text

def make_predictions():
    filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    
    if filepath:
        data = pd.read_excel(filepath)

        data['pr_txt'] = data['pr_txt'].apply(cleaning)

        count_tf_idf = joblib.load('tfidf_vectorizer.joblib')
        X = data['pr_txt']
        tfidf_data = count_tf_idf.transform(X)

        best_model_category = joblib.load('best_model_category.joblib')
        data['category_pred'] = best_model_category.predict(tfidf_data)

        X_text_tfidf = count_tf_idf.transform(data['pr_txt'])

        ohe = joblib.load('ohe_encoder.joblib')
        X_category_ohe = ohe.transform(data[['category_pred']])

        X = hstack([X_text_tfidf, X_category_ohe])

        model = joblib.load('model_rating.joblib')
        data['Уровень рейтинга'] = model.predict(X)
        
        data['Категория'] = data['category_pred']
        data.drop(columns=['category_pred'], inplace=True)
        
        output_filepath = filepath.replace(".xlsx", "_predictions.csv")
        data[['Id','Категория','Уровень рейтинга']].to_csv(output_filepath, sep=';', index=False)
        
        label.config(text="Предсказание выполнено!")

root = Tk()
root.title("Предсказание категории и уровня рейтинга")

btn = Button(root, text="Загрузите файл для предсказания", command=make_predictions)
btn.pack(pady=20)

label = Label(root, text="")
label.pack(pady=20)

root.mainloop()