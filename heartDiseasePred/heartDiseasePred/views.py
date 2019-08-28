from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .forms import predForm
from .predictor import predict

def home(request):
	if request.method == 'POST':
		form = predForm(request.POST)
		if form.is_valid():
			data = form.cleaned_data
			age = data['age']
			sex = data['sex']
			cp = data['cp']
			trestbps = data['trestbps']
			chol = data['chol']
			fbs = data['fbs']
			restecg = data['restecg']
			thalach = data['thalach']
			exang =  data['exang']
			oldpeak = data['oldpeak']
			slope = data['slope']
			ca = data['ca']
			thal = data['thal']
			res = predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
			if res == 1:
				return HttpResponseRedirect('/true/')
			elif res == -1:
				return HttpResponseRedirect('/error/')
			return HttpResponseRedirect('/false/')

	form = predForm()
	return render(request, 'heartDiseasePred/home.html', {'form': form})

def trueRes(request):
	return render(request, 'heartDiseasePred/true.html')

def falseRes(request):
	return render(request, 'heartDiseasePred/false.html')

def errRes(request):
	return render(request, 'heartDiseasePred/error.html')

def about(request):
	return render(request, 'heartDiseasePred/about.html')