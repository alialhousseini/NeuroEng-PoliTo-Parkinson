1. First, install the necessary packages:
```
pip install tensorflow==2.4.1
pip install django
pip install Pillow
```

2. Create a Django project and app:
```
django-admin startproject parkinsons
cd parkinsons
python manage.py startapp predictor
```

3. Move the `model.h5` file to the `predictor` directory.

4. In the `predictor` directory, create a `views.py` file with the following code:
```python
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'POST':
        # get the uploaded image from the form
        uploaded_file = request.FILES['document']
        # save the image to the media directory
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        # get the saved image path
        image_url = fs.url(name)
        # load the model
        model = load_model('model.h5')
        # load the image using keras image module
        img = image.load_img('.'+image_url, target_size=(100, 100))
        # convert the image to a numpy array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # normalize the image pixels
        x = x/255.0
        # make the prediction
        prediction = model.predict(x)[0][0]
        if prediction < 0.5:
            result = "You do not have Parkinson's disease."
        else:
            result = "You have Parkinson's disease."
        # render the result template with the prediction result
        return render(request, 'result.html', {'result': result})
```

5. In the `predictor` directory, create a `templates` directory and create two files: `home.html` and `result.html`.

`home.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Parkinson's Disease Predictor</title>
</head>
<body>
    <h1>Parkinson's Disease Predictor</h1>
    <form method="post" action="{% url 'result' %}" enctype="multipart/form-data">
        {% csrf_token %}
        <input type="file" name="document"><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
```

`result.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p>{{ result }}</p>
</body>
</html>
```

6. In the `parkinsons` directory, open the `urls.py` file and add the following code:
```python
from django.urls import path
from predictor.views import home, result

urlpatterns = [
    path('', home, name='home'),
    path('result/', result, name='result'),
]
```

7. Start the Django server using the following command:
```
python manage.py runserver
```

Now you can access the app in your browser at `http://127.0.0.1:8000/` and upload an image to get the prediction result.