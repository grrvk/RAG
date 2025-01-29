from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
import json

from chat.processors.simple_rag import generateAnswer, findFavourites, storeUserInformation


def chat_interface(request):
    return render(request, 'chat_interface.html')


def chat_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_query = data.get('query', '')

        if findFavourites(user_query):
            storeUserInformation(user_query)

        rag_response = generateAnswer(user_query)
        return JsonResponse({'response': rag_response})

