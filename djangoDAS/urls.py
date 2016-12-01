"""djangoDAS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from face_detector import views
from django.conf import settings
from django.conf.urls.static import static
from caffe_classification import views as views1

urlpatterns = [
    url(r'^face_detection/detect/$', views.detect, name='detect'),
    url(r'^admin/', admin.site.urls),
    url(r'^index/', views1.index, name='index'),
    url(r'^probability/', views1.probability, name='probability'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
