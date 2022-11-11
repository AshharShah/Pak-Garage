from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.template import loader
# Create your views here.

@csrf_exempt
def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

@csrf_exempt
def output(request):
    return render(request, 'output.html')