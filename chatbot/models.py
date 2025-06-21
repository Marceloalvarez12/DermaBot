# chatbot/models.py
import uuid
import os
from django.db import models
from django.utils import timezone

# --- Modelo Desease (Asumiendo que ya lo tienes unificado como discutimos) ---
class Desease(models.Model):
    abbreviation = models.CharField(
        max_length=20, blank=True, null=True,
        verbose_name="Abreviatura (Ej: salida CNN)"
    )
    name_desease = models.CharField(
        max_length=150, unique=True, 
        verbose_name="Nombre de la Enfermedad"
    )
    description = models.TextField(
        blank=True, null=True, 
        verbose_name="Descripción General (Opcional)"
    )
    short_description_for_llm = models.TextField(
        verbose_name="Descripción Breve para LLM (MUY CONCISA)",
        help_text="Máx 5-15 palabras clave. Ej: 'Acné: Granos, espinillas. Cara/pecho/espalda.'"
    )
    cnn_prediction_index = models.IntegerField(
        unique=True, null=True, blank=True, 
        verbose_name="Índice de Predicción CNN (si aplica)",
        help_text="El índice numérico (0, 1, ...) que tu CNN devuelve para esta enfermedad."
    )
    common_symptoms_list = models.TextField(
        verbose_name="Lista de Síntomas Comunes (Opcional)", blank=True, null=True,
        help_text="Síntomas típicos, separados por comas."
    )
    key_questions_to_ask = models.TextField(
        verbose_name="Preguntas Clave Específicas (Opcional)", blank=True, null=True,
        help_text="Preguntas que el bot podría hacer si sospecha esta enfermedad."
    )
    general_advice_non_medical = models.TextField(
        verbose_name="Consejo General No Médico (Opcional)", blank=True, null=True,
        help_text="Consejos generales (no tratamiento)."
    )

    def __str__(self):
        return self.name_desease

    class Meta:
        verbose_name = "Enfermedad (Base de Conocimiento)"
        verbose_name_plural = "Enfermedades (Base de Conocimiento)"
        ordering = ['name_desease']

# --- Modelo Conversation ---
class Conversation(models.Model):
    id = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False,
        verbose_name="ID de Conversación"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Fecha de Creación"
    )
    # user_identifier_in_session = models.CharField(max_length=100, blank=True, null=True, verbose_name="Identificador de Usuario (Sesión)")


    def __str__(self):
        return f"Conversación {str(self.id)[:8]}"

    class Meta:
        verbose_name = "Conversación del Chatbot"
        verbose_name_plural = "Conversaciones del Chatbot"
        ordering = ['-created_at']

# --- Modelo Message (sin cambios respecto a la última versión que te pasé arriba) ---
class Message(models.Model):
    conversation = models.ForeignKey(
        Conversation, 
        related_name='messages', 
        on_delete=models.CASCADE,
        verbose_name="Conversación"
    )
    content = models.TextField(
        verbose_name="Contenido del Mensaje de Texto", 
        blank=True, 
        null=True
    )
    is_bot = models.BooleanField(
        default=False, 
        verbose_name="¿Es del Bot?"
    )
    timestamp = models.DateTimeField(
        default=timezone.now,
        verbose_name="Marca de Tiempo"
    )
    image = models.ImageField(
        upload_to='chatbot_images/', 
        null=True, 
        blank=True,
        verbose_name="Imagen Adjunta (Opcional)"
    )
    cnn_predicted_desease = models.ForeignKey( 
        Desease, 
        verbose_name='Enfermedad Sugerida por CNN (Opcional)',
        null=True, 
        blank=True, 
        on_delete=models.SET_NULL
    )
    cnn_confidence = models.FloatField(
        null=True, 
        blank=True, 
        verbose_name="Confianza CNN (%)"
    )

    class Meta:
        ordering = ['timestamp']
        verbose_name = "Mensaje de Chat"
        verbose_name_plural = "Mensajes de Chat"

    def __str__(self):
        actor = "DermaBot" if self.is_bot else "Usuario"
        local_timestamp = timezone.localtime(self.timestamp)
        base_str = f"[{local_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {actor}: "
        if self.image and self.image.name:
            image_name = os.path.basename(self.image.name)
            base_str += f"[Imagen: {image_name}] "
        if self.content:
            base_str += self.content[:50]
        if not self.content and not (self.image and self.image.name):
             base_str += "[Mensaje sin contenido explícito]"
        if self.cnn_predicted_desease:
            conf_str = f" ({self.cnn_confidence:.1f}%)" if self.cnn_confidence is not None else ""
            base_str += f" (CNN Sugiere: {self.cnn_predicted_desease.name_desease}{conf_str})"
        return base_str + ("..." if len(self.content or "") > 50 else "")

# --- NUEVO MODELO: MedicalSummary ---
class MedicalSummary(models.Model):
    conversation = models.OneToOneField(
        Conversation,
        on_delete=models.CASCADE,
        primary_key=True, 
        related_name='medical_summary',
        verbose_name="Conversación Asociada"
    )
    # Campo general para el resumen generado por el LLM
    summary_text_generated_by_llm = models.TextField(
        verbose_name="Resumen de Ficha Preliminar (Generado por IA)",
        blank=True, null=True,
        help_text="Texto completo del resumen estructurado generado por el LLM."
    )
    
    # Campos específicos extraídos del resumen (opcional, si el parsing es robusto)
    main_complaint = models.TextField(blank=True, null=True, verbose_name="Motivo Principal de Consulta")
    symptoms_reported = models.TextField(blank=True, null=True, verbose_name="Síntomas Reportados")
    location_of_symptoms = models.CharField(max_length=255, blank=True, null=True, verbose_name="Localización de Síntomas")
    duration_of_symptoms = models.CharField(max_length=255, blank=True, null=True, verbose_name="Duración de Síntomas")
    aggravating_factors = models.TextField(blank=True, null=True, verbose_name="Factores Agravantes")
    alleviating_factors = models.TextField(blank=True, null=True, verbose_name="Factores de Alivio")
    previous_history = models.TextField(blank=True, null=True, verbose_name="Antecedentes Relevantes")
    image_analysis_summary_from_cnn = models.TextField(blank=True, null=True, verbose_name="Resumen del Análisis de Imagen (CNN)")
    tentative_orientation_by_llm = models.TextField(blank=True, null=True, verbose_name="Orientación Tentativa del LLM (Texto)")

    last_updated = models.DateTimeField(auto_now=True, verbose_name="Última Actualización del Resumen")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de Creación del Resumen")

    def __str__(self):
        return f"Resumen para Conversación {str(self.conversation_id)[:8]}"

    class Meta:
        verbose_name = "Resumen de Conversación (Ficha Preliminar IA)"
        verbose_name_plural = "Resúmenes de Conversación (Fichas Preliminares IA)"
        ordering = ['-created_at']