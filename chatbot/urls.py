# chatbot/urls.py
from django.urls import path
from .views import ChatHomeView, ChatWindowView, StartNewChatSessionView

app_name = 'chatbot'

urlpatterns = [
    path('', ChatHomeView.as_view(), name='chat_home'),
    path('new/', StartNewChatSessionView.as_view(), name='start_new_session'),
    path('session/<uuid:conversation_id>/', ChatWindowView.as_view(), name='chat_window'),
]