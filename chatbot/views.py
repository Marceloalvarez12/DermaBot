# chatbot/views.py
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.http import HttpResponse, Http404 # Asegúrate que Http404 está importado
from django.template.loader import get_template
from django.core.paginator import Paginator
from django.utils import timezone # Para el nombre del archivo PDF

from .models import Conversation, Message, Desease, MedicalSummary
from .forms import MessageForm
# ASEGÚRATE DE TENER LOS __init__.py EN LA CARPETA services
from .services.openai_agent_service import DermaBotAgent
from .services.cnn_service import CNNProcessor

# Instanciación de los servicios (Singleton pattern)
try:
    derma_agent_llm = DermaBotAgent.get_instance()
except Exception as e:
    print(f"!!!!!!!! ERROR CRÍTICO al instanciar DermaBotAgent: {e} !!!!!!!!")
    derma_agent_llm = None # Manejar la ausencia del agente en las vistas

try:
    # Usar la instanciación normal si CNNProcessor no tiene get_instance()
    # o cnn_image_processor = CNNProcessor.get_instance() si lo implementaste
    cnn_image_processor = CNNProcessor() 
except Exception as e:
    print(f"!!!!!!!! ERROR CRÍTICO al instanciar CNNProcessor: {e} !!!!!!!!")
    cnn_image_processor = None # Manejar la ausencia del procesador en las vistas


class ChatHomeView(View):
    def get(self, request):
        conversation_id_str = request.session.get('chatbot_conversation_id')
        conversation = None
        if conversation_id_str:
            try:
                conversation = Conversation.objects.get(id=uuid.UUID(conversation_id_str))
            except (Conversation.DoesNotExist, ValueError):
                conversation = None # Forzar creación si el ID no es válido o no existe
        
        if not conversation:
            conversation = Conversation.objects.create()
            request.session['chatbot_conversation_id'] = str(conversation.id)
            print(f"--- VIEW DEBUG: ChatHomeView - Nueva conversación creada o ID de sesión establecido: {conversation.id}")
        
        return redirect('chatbot:chat_window', conversation_id=conversation.id)

class StartNewChatSessionView(View):
    def get(self, request):
        # Crea una nueva conversación siempre
        new_conversation = Conversation.objects.create()
        # Actualiza el ID en la sesión
        request.session['chatbot_conversation_id'] = str(new_conversation.id)
        print(f"--- VIEW DEBUG: StartNewChatSessionView - Nueva conversación ID: {new_conversation.id}")
        return redirect('chatbot:chat_window', conversation_id=new_conversation.id)

class ChatWindowView(View):
    template_name = 'chatbot/chat_window.html'
    
    def get(self, request, conversation_id): # conversation_id es un objeto UUID pasado por la URL
        conversation = get_object_or_404(Conversation, id=conversation_id)
        
        if request.session.get('chatbot_conversation_id') != str(conversation.id):
            request.session['chatbot_conversation_id'] = str(conversation.id)

        messages = conversation.messages.all().order_by('timestamp')
        form = MessageForm()
        
        user_identifier = request.session.get('chatbot_user_nickname', f"Usuario_{str(conversation.id)[:8]}")
        
        medical_summary_exists = MedicalSummary.objects.filter(conversation=conversation).exists()
        
        context = { 
            'conversation': conversation,
            'messages': messages,
            'form': form,
            'user_identifier': user_identifier,
            'medical_summary_exists': medical_summary_exists,
            'conversation_id_str': str(conversation.id) 
        }
        return render(request, self.template_name, context)

    def post(self, request, conversation_id): 
        if derma_agent_llm is None:
            messages = Message.objects.filter(conversation_id=conversation_id).order_by('timestamp') # Recuperar mensajes existentes
            form = MessageForm(request.POST, request.FILES)
            conversation = get_object_or_404(Conversation, id=conversation_id) # Necesitas la conversación para el contexto
            user_identifier = request.session.get('chatbot_user_nickname', f"Usuario_{str(conversation.id)[:8]}")
            medical_summary_exists = MedicalSummary.objects.filter(conversation=conversation).exists()
            
            context = { 
                'conversation': conversation, 'messages': messages, 'form': form, 
                'user_identifier': user_identifier, 'medical_summary_exists': medical_summary_exists,
                'conversation_id_str': str(conversation.id),
                'error_message': "Servicio de chat no disponible debido a un problema de inicialización del agente."
            }
            return render(request, self.template_name, context, status=503)


        conversation = get_object_or_404(Conversation, id=conversation_id)
        form = MessageForm(request.POST, request.FILES) 
        user_identifier = request.session.get('chatbot_user_nickname', f"Usuario_{str(conversation.id)[:8]}")

        if form.is_valid():
            user_input_text = form.cleaned_data.get('user_input')
            uploaded_image_file = form.cleaned_data.get('image_upload')

            user_message_obj = Message(
                conversation=conversation,
                content=user_input_text, 
                is_bot=False
            )
            
            input_for_llm = user_input_text if user_input_text else ""
            cnn_prediction_info_for_llm = "" 

            if uploaded_image_file:
                if cnn_image_processor is None:
                    user_message_obj.content = user_input_text if user_input_text else "[Imagen subida, pero procesador CNN no disponible]"
                    user_message_obj.image = uploaded_image_file
                    user_message_obj.save()
                    Message.objects.create(conversation=conversation, content="Lo siento, el servicio de análisis de imágenes no está disponible actualmente.", is_bot=True)
                    return redirect('chatbot:chat_window', conversation_id=conversation.id)

                print(f"--- VIEW DEBUG: ChatWindowView POST - Procesando imagen subida: {uploaded_image_file.name}")
                user_message_obj.image = uploaded_image_file 

                predicted_desease_obj, confidence_percent = cnn_image_processor.predict_from_image_file(uploaded_image_file)
                
                user_message_obj.cnn_confidence = confidence_percent
                if predicted_desease_obj:
                    user_message_obj.cnn_predicted_desease = predicted_desease_obj
                    cnn_prediction_info_for_llm = (
                        f"Contexto de imagen: El análisis preliminar de la imagen subida por el usuario sugiere "
                        f"que podría estar relacionado con '{predicted_desease_obj.name_desease}' "
                        f"(confianza de la CNN: {confidence_percent:.1f}%). "
                        f"Considera esta información en tu diálogo y orientación."
                    )
                    print(f"--- VIEW DEBUG: Predicción CNN: {predicted_desease_obj.name_desease}, Confianza: {confidence_percent:.1f}%")
                else:
                    cnn_prediction_info_for_llm = (
                        "Contexto de imagen: El usuario subió una imagen, pero el análisis preliminar de la CNN "
                        "no pudo determinar una condición específica de su lista de referencia. "
                        "Procede con preguntas generales sobre la apariencia si es necesario."
                    )
                    print("--- VIEW DEBUG: Predicción CNN: No se pudo determinar una condición específica.")
            
            user_message_obj.save()

            if user_input_text and cnn_prediction_info_for_llm:
                final_input_for_llm = f"{cnn_prediction_info_for_llm} El usuario también comentó: '{user_input_text}'"
            elif cnn_prediction_info_for_llm: 
                final_input_for_llm = cnn_prediction_info_for_llm
            elif user_input_text: 
                final_input_for_llm = user_input_text
            else: 
                final_input_for_llm = "El usuario no proporcionó entrada." 
            
            print(f"--- VIEW DEBUG: ChatWindowView POST - Input final para LLM: '{final_input_for_llm}'")
            
            bot_response_content_for_user = derma_agent_llm.get_response(
                user_input=final_input_for_llm,
                conversation_id=str(conversation.id),
                user_identifier=user_identifier
            )
            
            Message.objects.create(conversation=conversation, content=bot_response_content_for_user, is_bot=True)
            
            return redirect('chatbot:chat_window', conversation_id=conversation.id)
        
        messages = conversation.messages.all().order_by('timestamp')
        medical_summary_exists = MedicalSummary.objects.filter(conversation=conversation).exists()
        context = {
            'conversation': conversation,
            'messages': messages,
            'form': form, 
            'user_identifier': user_identifier,
            'medical_summary_exists': medical_summary_exists,
            'conversation_id_str': str(conversation.id)
        }
        return render(request, self.template_name, context)


class MedicalSummaryDetailView(View):
    html_template_name = 'chatbot/medical_summary_detail.html'

    def get(self, request, summary_id_uuid): # summary_id_uuid es el UUID de la conversación (PK del MedicalSummary)
        print(f"--- VIEW DEBUG: MedicalSummaryDetailView GET - Intentando obtener summary con PK (conversation_id UUID): {summary_id_uuid}")
        
        ################################################################################
        # CORRECCIÓN DEL FieldError:
        # El modelo MedicalSummary tiene 'conversation' (que es un OneToOneField a Conversation
        # y también primary_key=True). Por lo tanto, su PK es el ID de la conversación.
        # 'pk' es un alias para el campo de clave primaria.
        ################################################################################
        medical_summary = get_object_or_404(MedicalSummary, pk=summary_id_uuid)
        
        print(f"--- VIEW DEBUG: MedicalSummaryDetailView - Resumen médico recuperado ID (que es conversation_id): {medical_summary.pk}") # .pk es el valor de la PK
        if medical_summary.conversation:
             print(f"--- VIEW DEBUG: MedicalSummaryDetailView - Conversación asociada ID: {medical_summary.conversation.id}")
        else:
             # Esto no debería pasar si la PK es la conversación
             print("--- VIEW WARNING: MedicalSummaryDetailView - Resumen médico NO tiene conversación asociada explícitamente (revisar modelo).") 
        
        # Imprimir los campos que la plantilla espera para depuración
        print(f"--- VIEW DEBUG: Resumen Texto (del LLM): '{getattr(medical_summary, 'summary_text_generated_by_llm', 'CAMPO NO ENCONTRADO')[:100]}...'")
        print(f"--- VIEW DEBUG: Orientación Tentativa (del LLM): '{getattr(medical_summary, 'tentative_orientation_by_llm', 'CAMPO NO ENCONTRADO')[:100]}...'")
        print(f"--- VIEW DEBUG: Queja Principal: '{getattr(medical_summary, 'main_complaint', 'CAMPO NO ENCONTRADO')}'")
        # Puedes añadir más prints para otros campos si es necesario

        context = {
            'medical_summary': medical_summary,
            'conversation': medical_summary.conversation, 
            'page_title': 'Ficha Médica Preliminar'
        }
        return render(request, self.html_template_name, context)


class MedicalSummaryPDFView(View):
    pdf_template_name = 'chatbot/medical_summary_pdf.html'

    def get(self, request, summary_id_uuid): # summary_id_uuid es el UUID de la conversación (PK del MedicalSummary)
        print(f"--- VIEW DEBUG: MedicalSummaryPDFView GET - Intentando generar PDF para summary con PK: {summary_id_uuid}")
        medical_summary = get_object_or_404(MedicalSummary, pk=summary_id_uuid) # Usar pk aquí también
        
        template = get_template(self.pdf_template_name)
        context_pdf = {
            'summary': medical_summary, # La plantilla PDF que me pasaste usa 'summary'
            'conversation': medical_summary.conversation,
            'current_date': timezone.now() 
        }
        html = template.render(context_pdf)
        
        response = HttpResponse(content_type='application/pdf')
        filename = f"resumen_dermabot_conv_{str(medical_summary.conversation.id)[:8]}_sum_{str(medical_summary.pk)[:8]}.pdf"
        response['Content-Disposition'] = f'inline; filename="{filename}"'

        try:
            from xhtml2pdf import pisa 
            
            pisa_status = pisa.CreatePDF(html, dest=response)
            if pisa_status.err:
                print(f"!!!!!!!! ERROR generando PDF con xhtml2pdf: {pisa_status.err} !!!!!!!!")
                return HttpResponse('Error interno al generar el PDF.', status=500)
            
            print(f"--- VIEW DEBUG: MedicalSummaryPDFView - PDF generado exitosamente para summary PK: {medical_summary.pk}")
            return response
            
        except ImportError:
            print("!!!!!!!! ADVERTENCIA: 'xhtml2pdf' no está instalada. Funcionalidad PDF no disponible. !!!!!!!!")
            return HttpResponse("Funcionalidad PDF no disponible (Falta la librería 'xhtml2pdf'). Por favor, instálala.", status=501)
        except Exception as e:
            print(f"!!!!!!!! ERROR inesperado generando PDF: {e} !!!!!!!!")
            return HttpResponse(f"Error inesperado al generar el PDF: {e}", status=500)


class ConversationHistoryListView(View):
    template_name = 'chatbot/conversation_history_list.html'
    paginate_by = 10

    def get(self, request):
        summaries_list = MedicalSummary.objects.select_related('conversation').order_by('-conversation__created_at')
        
        paginator = Paginator(summaries_list, self.paginate_by)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        print(f"--- VIEW DEBUG: ConversationHistoryListView - Mostrando página {page_obj.number} de {paginator.num_pages}")
        
        context = {
            'page_obj': page_obj, 
            'is_paginated': page_obj.has_other_pages(),
            'page_title': 'Historial de Conversaciones con Resumen'
        }
        return render(request, self.template_name, context)