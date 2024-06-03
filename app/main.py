from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from joblib import load
from pydantic import BaseModel

# Load the model
spam_clf = load(open('./spam_detector_model.pkl', 'rb'))

# Load vectorizer
vectorizer = load(open('./vectorizer.pickle', 'rb'))

# Initialize an instance of FastAPI
app = FastAPI()

# Define the input data model
class Message(BaseModel):
    text_message: str

# Define the default route 
# @app.get("/")
# async def read_index():
#     return FileResponse('static/index.html')

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}

# # Define the route to the spam predictor
# @app.post("/predict_sentiment")
# async def predict_sentiment(message: Message):
#     text_message = message.text_message

#     if(not(text_message)):
#         raise HTTPException(status_code=400, detail="Please provide a valid text message")

#     prediction = spam_clf.predict(vectorizer.transform([text_message]))

#     if prediction[0] == 0:
#         polarity = "Ham"
#     else:
#         polarity = "Spam"
        
#     return {
#         "text_message": text_message,
#         "sentiment_polarity": polarity
#     }

# Define the route to the sentiment predictor
@app.post("/predict_sentiment")
def predict_sentiment(text_message):

    polarity = ""

    if(not(text_message)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid text message")

    prediction = spam_clf.predict(vectorizer.transform([text_message]))

    if(prediction[0] == 0):
        polarity = "Ham"

    elif(prediction[0] == 1):
        polarity = "Spam"
        
    return {
            "text_message": text_message, 
            "sentiment_polarity": polarity
           }