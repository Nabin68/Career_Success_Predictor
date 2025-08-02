from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib

dataset=pd.read_csv(r"D:\ML_Projects\Career_Success_Predictor\datasets\Salary_prediction.csv")
print(dataset.head(5))

#filling null values
dataset.fillna({
    "Age":dataset["Age"].median(),
    "Gender":dataset["Gender"].mode()[0],
    "Education Level":dataset["Education Level"].mode()[0],
    "Job Title":dataset["Job Title"].mode()[0],
    "Years of Experience":dataset["Years of Experience"].median(),
    "Salary":dataset["Salary"].median(),
},inplace=True)

def extract_level(title):
    title=title.lower()
    if "junior"  in title:
        return "Junior"
    elif "senior" in title or "principal"in title:
        return "Senior"
    elif "lead" in title or "manager" in title or"director" in title or "chief" in title or "vp" in title:
        return "Senior"
    else:
        return "Mid"
    
dataset["Carrer_Level"]=dataset["Job Title"].apply(extract_level)
dataset.drop(columns="Job Title",inplace=True)

#encoding categorical data
categorical_cols = ["Gender", "Education Level", "Carrer_Level"]
ohe = OneHotEncoder(drop="first", sparse_output=False)
encoded_array = ohe.fit_transform(dataset[categorical_cols])
encoded_col_names = ohe.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=dataset.index)
dataset = dataset.drop(columns=categorical_cols)
dataset = pd.concat([dataset, encoded_df], axis=1)
dataset.drop_duplicates(inplace=True)

#scaling features
scaler=StandardScaler()
scale_cols = ["Age", "Years of Experience"]
scaler = StandardScaler()
dataset[scale_cols] = scaler.fit_transform(dataset[scale_cols])

#putting salary column to last
salary_col=dataset.pop("Salary")
dataset["Salary"]=salary_col
print(dataset.head())

x=dataset.iloc[:,:-1]
y=dataset["Salary"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

# Save model and transformers
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(ohe, 'encoder.pkl')