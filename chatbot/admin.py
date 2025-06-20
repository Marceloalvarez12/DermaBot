# chatbot/admin.py
from django.contrib import admin
from .models import Conversation, Message, Desease # <<--- CAMBIO AQUÍ: KnownDesease a Desease
from django.utils.html import format_html # Para la vista previa de la imagen (si la usas)

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at')
    readonly_fields = ('id', 'created_at')
    list_per_page = 20

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
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
    # readonly_fields son para la vista de detalle/edición, no para list_display
    # Si quieres que todos sean readonly en el form de edición:
    readonly_fields = ('conversation', 'content', 'is_bot', 'timestamp', 
                       'image_display_for_detail', # Usar un método separado para el form
                       'cnn_predicted_desease', 'cnn_confidence') 
    list_per_page = 20
    
    fieldsets = (
        (None, {'fields': ('conversation', 'timestamp', 'is_bot')}),
        ('Contenido del Mensaje', {'fields': ('content', 'image_display_for_detail', 'image')}),
        ('Análisis CNN (si aplica)', {'fields': ('cnn_predicted_desease', 'cnn_confidence')}),
    )

    def conversation_id_short(self, obj):
        return str(obj.conversation.id)[:8]
    conversation_id_short.short_description = 'ID Conversación'

    def timestamp_formatted(self, obj):
        from django.utils import timezone # Importar aquí o al inicio del archivo
        return timezone.localtime(obj.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    timestamp_formatted.short_description = 'Marca de Tiempo'
    timestamp_formatted.admin_order_field = 'timestamp'

    def content_preview(self, obj):
        if obj.content:
            return (obj.content[:75] + '...') if len(obj.content) > 75 else obj.content
        return "(Sin texto)"
    content_preview.short_description = 'Contenido (Texto)'

    def cnn_prediction_preview(self, obj):
        if obj.cnn_predicted_desease:
            conf_str = f" ({obj.cnn_confidence:.1f}%)" if obj.cnn_confidence is not None else ""
            return f"{obj.cnn_predicted_desease.name_desease}{conf_str}"
        return "N/A"
    cnn_prediction_preview.short_description = 'Predicción CNN'

    def image_preview(self, obj): # Para list_display
        if obj.image and hasattr(obj.image, 'url'):
            return format_html('<img src="{}" style="max-height: 50px; max-width: 50px;" />', obj.image.url)
        return "(Sin imagen)"
    image_preview.short_description = 'Img.'
    
    def image_display_for_detail(self, obj): # Para fieldsets (vista de detalle/edición)
        if obj.image and hasattr(obj.image, 'url'):
            return format_html('<a href="{0}" target="_blank"><img src="{0}" style="max-height: 200px; max-width: 200px;" /></a>', obj.image.url)
        return "(Sin imagen)"
    image_display_for_detail.short_description = 'Imagen Actual'


@admin.register(Desease) # <<--- CAMBIO AQUÍ: KnownDesease a Desease
class DeseaseAdmin(admin.ModelAdmin): # <<--- CAMBIO AQUÍ: Renombrar la clase Admin
    list_display = (
        'name_desease',
        'abbreviation',
        'short_description_for_llm_preview',
        'cnn_prediction_index',
        # 'common_symptoms_list_preview' # Puedes añadir más si quieres
    )
    search_fields = ('name_desease', 'abbreviation')
    list_filter = ('abbreviation',) 
    list_editable = (
        'abbreviation', 
        'cnn_prediction_index', 
        # 'short_description_for_llm' # Editar TextFields en la lista puede ser incómodo
    ) 
    list_per_page = 25

    fieldsets = ( # Para organizar la vista de edición de Desease
        (None, {'fields': ('name_desease', 'abbreviation', 'description')}),
        ('Información para la CNN', {'fields': ('cnn_prediction_index',)}),
        ('Información para el LLM (DermaBot)', {'fields': ('short_description_for_llm', 'common_symptoms_list', 'key_questions_to_ask', 'general_advice_non_medical')}),
    )

    def short_description_for_llm_preview(self, obj):
        if obj.short_description_for_llm:
            return (obj.short_description_for_llm[:75] + '...') if len(obj.short_description_for_llm) > 75 else obj.short_description_for_llm
        return "-"
    short_description_for_llm_preview.short_description = 'Desc. Breve (LLM)'

    # def common_symptoms_list_preview(self, obj):
    #     if obj.common_symptoms_list:
    #         return (obj.common_symptoms_list[:75] + '...') if len(obj.common_symptoms_list) > 75 else obj.common_symptoms_list
    #     return "-"
    # common_symptoms_list_preview.short_description = 'Síntomas Comunes'