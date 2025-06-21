# chatbot/urls.py
from django.urls import path
from .views import (
    ChatHomeView, ChatWindowView, StartNewChatSessionView,
    MedicalSummaryDetailView,
    MedicalSummaryPDFView, # Si tienes una vista separada para el PDF
    ConversationHistoryListView )

app_name = 'chatbot'

urlpatterns = [
    path('', ChatHomeView.as_view(), name='chat_home'),
    path('new/', StartNewChatSessionView.as_view(), name='start_new_session'),
    path('session/<uuid:conversation_id>/', ChatWindowView.as_view(), name='chat_window'),
    path('summary/<uuid:summary_id_uuid>/', MedicalSummaryDetailView.as_view(), name='medical_summary_detail'),
    path('summary/<uuid:summary_id_uuid>/pdf/', MedicalSummaryPDFView.as_view(), name='medical_summary_pdf'), # URL para el PDF
    path('historial/', ConversationHistoryListView.as_view(), name='conversation_history'),]
