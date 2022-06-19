import uvicorn
from fastapi import FastAPI
import pickle
app = FastAPI()

#Correct File address
pickle_in = open("KNN_Classifier.pkl","rb")
classifier = pickle.load(pickle_in)


@app.get("/")
def index():
    return {"message":"Hello World"}

@app.get("/predict")
def get_name(variance:float,skewness:float,curtosis:float,entropy:float):
    prediction =classifier.predict([[variance,skewness,curtosis,entropy]])
    if prediction[0] > 0:
        prediction = "Note is Fake"
    else:
        prediction = "Note is Real"
    return {
        "prediction":prediction
    }
if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)