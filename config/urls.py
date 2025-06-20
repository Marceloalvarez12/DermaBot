# config/urls.py
from django.contrib import admin
from django.urls import path, include # Asegúrate que include esté importado
from django.shortcuts import redirect # Para la redirección de la raíz

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dermabot/', include('chatbot.urls')), # Prefijo 'dermabot/' para nuestra app
    # Redirigir la raíz del sitio al inicio del chat
    path('', lambda request: redirect('chatbot:chat_home', permanent=False)),
]