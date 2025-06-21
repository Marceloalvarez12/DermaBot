# chatbot/admin.py
from django.contrib import admin
from .models import Conversation, Message, Desease, MedicalSummary # <<--- AÑADIR MedicalSummary
from django.utils.html import format_html

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id_short', 'created_at', 'has_summary') # Añadido has_summary
    readonly_fields = ('id', 'created_at')
    list_per_page = 20

    def id_short(self, obj):
        return str(obj.id)[:8]
    id_short.short_description = 'ID Corto'

    def has_summary(self, obj):
        return hasattr(obj, 'medical_summary') and obj.medical_summary is not None
    has_summary.boolean = True
    has_summary.short_description = '¿Tiene Resumen?'


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    # ... (como lo tenías antes está bien, sin cambios necesarios aquí por MedicalSummary) ...
    list_display = (
        'conversation_id_short', 
        'is_bot', 
        'image_preview', 
        'content_preview', 
        'cnn_prediction_preview', 
        'timestamp_formatted'
    )
    list_filter = ('is_bot', 'timestamp', 'conversation__id')
    search_fields = ('content', 'conversation__id__iexact')
    readonly_fields = ('conversation', 'content', 'is_bot', 'timestamp', 
                       'image_display_for_detail', 
                       'cnn_predicted_desease', 'cnn_confidence') 
    list_per_page = 20
    fieldsets = (
        (None, {'fields': ('conversation', 'timestamp', 'is_bot')}),
        ('Contenido del Mensaje', {'fields': ('content', 'image_display_for_detail', 'image')}),
        ('Análisis CNN (si aplica)', {'fields': ('cnn_predicted_desease', 'cnn_confidence')}),
    )
    def conversation_id_short(self, obj): return str(obj.conversation.id)[:8]
    conversation_id_short.short_description = 'ID Conversación'
    def timestamp_formatted(self, obj):
        from django.utils import timezone
        return timezone.localtime(obj.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    timestamp_formatted.short_description = 'Marca de Tiempo'
    timestamp_formatted.admin_order_field = 'timestamp'
    def content_preview(self, obj):
        if obj.content: return (obj.content[:75] + '...') if len(obj.content) > 75 else obj.content
        return "(Sin texto)"
    content_preview.short_description = 'Contenido (Texto)'
    def cnn_prediction_preview(self, obj):
        if obj.cnn_predicted_desease:
            conf_str = f" ({obj.cnn_confidence:.1f}%)" if obj.cnn_confidence is not None else ""
            return f"{obj.cnn_predicted_desease.name_desease}{conf_str}"
        return "N/A"
    cnn_prediction_preview.short_description = 'Predicción CNN'
    def image_preview(self, obj):
        if obj.image and hasattr(obj.image, 'url'): return format_html('<img src="{}" style="max-height: 50px; max-width: 50px;" />', obj.image.url)
        return "(Sin imagen)"
    image_preview.short_description = 'Img.'
    def image_display_for_detail(self, obj):
        if obj.image and hasattr(obj.image, 'url'): return format_html('<a href="{0}" target="_blank"><img src="{0}" style="max-height: 200px; max-width: 200px;" /></a>', obj.image.url)
        return "(Sin imagen)"
    image_display_for_detail.short_description = 'Imagen Actual'


@admin.register(Desease)
class DeseaseAdmin(admin.ModelAdmin):
    # ... (como lo tenías antes está bien) ...
    list_display = ( 'name_desease', 'abbreviation', 'short_description_for_llm_preview', 'cnn_prediction_index', )
    search_fields = ('name_desease', 'abbreviation')
    list_filter = ('abbreviation',)
    list_editable = ('abbreviation', 'cnn_prediction_index', ) 
    list_per_page = 25
    fieldsets = (
        (None, {'fields': ('name_desease', 'abbreviation', 'description')}),
        ('Información para la CNN', {'fields': ('cnn_prediction_index',)}),
        ('Información para el LLM (DermaBot)', {'fields': ('short_description_for_llm', 'common_symptoms_list', 'key_questions_to_ask', 'general_advice_non_medical')}),
    )
    def short_description_for_llm_preview(self, obj):
        if obj.short_description_for_llm: return (obj.short_description_for_llm[:75] + '...') if len(obj.short_description_for_llm) > 75 else obj.short_description_for_llm
        return "-"
    short_description_for_llm_preview.short_description = 'Desc. Breve (LLM)'


# --- NUEVO REGISTRO PARA MedicalSummary ---
@admin.register(MedicalSummary)
class MedicalSummaryAdmin(admin.ModelAdmin):
    list_display = ('conversation_id_short', 'created_at_summary', 'last_updated_summary', 'main_complaint_preview')
    readonly_fields = ('conversation', 'created_at', 'last_updated', 
                       'summary_text_generated_by_llm', 'main_complaint', 'symptoms_reported', 
                       'location_of_symptoms', 'duration_of_symptoms', 'aggravating_factors', 
                       'alleviating_factors', 'previous_history', 'image_analysis_summary_from_cnn',
                       'tentative_orientation_by_llm')
    search_fields = ('conversation__id__iexact', 'summary_text_generated_by_llm')
    list_per_page = 20

    def conversation_id_short(self, obj):
        return str(obj.conversation_id)[:8] # Accede al conversation_id directamente porque es la PK
    conversation_id_short.short_description = 'ID Conversación'

    def created_at_summary(self, obj):
        return obj.created_at.strftime("%Y-%m-%d %H:%M")
    created_at_summary.short_description = 'Resumen Creado'
    created_at_summary.admin_order_field = 'created_at'
    
    def last_updated_summary(self, obj):
        return obj.last_updated.strftime("%Y-%m-%d %H:%M")
    last_updated_summary.short_description = 'Resumen Actualizado'
    last_updated_summary.admin_order_field = 'last_updated'

    def main_complaint_preview(self, obj):
        if obj.main_complaint:
            return (obj.main_complaint[:75] + '...') if len(obj.main_complaint) > 75 else obj.main_complaint
        return "N/A"
    main_complaint_preview.short_description = 'Motivo Principal'