import sys
from django.http import HttpResponse
from django.shortcuts import render
import json
from django.http import JsonResponse

from chat.processors.rag_processor import searchSimilar


def index(request):
    return HttpResponse("Trial index view")


def chat_interface(request):
    return render(request, 'chat_interface.html')


def chat_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_query = data.get('query', '')
        rag_response = searchSimilar(user_query)
        return JsonResponse({'response': rag_response})

