from django.shortcuts import render, HttpResponse, redirect
from .models import LdaInput
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create your views here.


def adminhome(request):
    if request.method == 'POST':
        getusername = request.POST.get('username')
        getpassword = request.POST.get('password')
        if getusername == 'admin' and getpassword == '4321':
            return redirect('/view')

        else:
            return HttpResponse('Invalid ')
    return render(request, 'adminhome.html')


def input(request):
    if request.method == 'POST':
        s_length = request.POST.get('s_length')
        s_width = request.POST.get('s_width')
        p_length = request.POST.get('p_length')
        p_width = request.POST.get('p_width')
        target = request.POST.get('target')
        species = request.POST.get('species')
        users = LdaInput()
        users.s_length = s_length
        users.s_width = s_width
        users.p_length = p_length
        users.p_width = p_width
        users.target = target
        users.species = species
        users.save()
    return render(request, 'input.html')

def viewdata(request):
    details = LdaInput.objects.all()
    return render(request, 'viewdata.html', {'value': details})


# load iris dataset
iris = datasets.load_iris()

# convert dataset to pandas DataFrame
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

# view first six rows of DataFrame
df.head()

# find how many total observations are in dataset
len(df.index)

# define predictor and response variables
X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['species']

# Fit the LDA model
model = LinearDiscriminantAnalysis()
model.fit(X, y)

# Define method to evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))

# define new observation
new = [5, 3, 1, .4]

# predict which class the new observation belongs to
model.predict([new])

# define data to plot
X = iris.data
y = iris.target
model = LinearDiscriminantAnalysis()
data_plot = model.fit(X, y).transform(X)
target_names = iris.target_names

# create LDA plot
plt.figure()
colors = ['red', 'green', 'blue']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
                label=target_name)

# add legend to plot
plt.legend(loc='best', shadow=False, scatterpoints=1)

# display LDA plot
plt.show()



