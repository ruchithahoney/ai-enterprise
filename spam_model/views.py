from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
import json
import joblib
import pickle
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('multinomial_nb_spam_model.joblib')
loaded_vectorizer = joblib.load('spam_vectorizer.joblib')

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # Transform the single input using the same vectorizer
        single_input_transformed = loaded_vectorizer.transform([data.get('text')])

        # Make prediction on the single input using the loaded model
        predicted_class = model.predict(single_input_transformed)
        return JsonResponse({'predicted_class': predicted_class[0]})
