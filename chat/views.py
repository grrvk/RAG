import sys
from django.http import HttpResponse
from django.shortcuts import render
import json
from django.http import JsonResponse

from chat.processors.faiss_processor import storeUserInformation, findFavourites, searchSimilar, generateAnswer


def index(request):
    return HttpResponse("Trial index view")


def chat_interface(request):
    return render(request, 'chat_interface.html')


def chat_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_query = data.get('query', '')

        if findFavourites(user_query):
            storeUserInformation(user_query)

        user_context = searchSimilar(user_query)
        combined_request = "\n".join(user_context + [user_query])

        rag_response = generateAnswer(combined_request)
        return JsonResponse({'response': rag_response})

