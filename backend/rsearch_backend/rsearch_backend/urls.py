"""rsearch_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rsearch_backend import backend
from rsearch_backend import init_db
urlpatterns = [
    path('admin/', admin.site.urls),
    path('query/', backend.query),
    path('insertData/', backend.insert_data),
    path('removeData/', backend.remove_data),
    path('insertDataset/', backend.insert_dataset),
    #path('removeDataset/', backend.remove_dataset),
    path('get_image/', backend.get_image),
    path('queryDataset/', backend.query_dataset),
    path('selectDataset/', backend.select_dataset),
    path('init_db/', init_db.init_db),

]
