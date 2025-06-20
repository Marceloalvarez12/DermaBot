# chatbot/views.py
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from .models import Conversation, Message, Desease
from .forms import MessageForm
# Ajusta estas importaciones a la nueva ubicación en la carpeta 'services'
from .services.openai_agent_service import DermaBotAgent 
from .services.cnn_service import CNNProcessor      


derma_agent_llm = DermaBotAgent.get_instance()
cnn_image_processor = CNNProcessor()

class ChatHomeView(View):
    def get(self, request):
        conversation_id_str = request.session.get('chatbot_conversation_id')
        conversation = None
        if conversation_id_str:
            try:
                conversation = Conversation.objects.get(id=uuid.UUID(conversation_id_str))
            except (Conversation.DoesNotExist, ValueError):
                conversation = None
        if not conversation:
            conversation = Conversation.objects.create()
            request.session['chatbot_conversation_id'] = str(conversation.id)
        return redirect('chatbot:chat_window', conversation_id=conversation.id)

class ChatWindowView(View):
    template_name = 'chatbot/chat_window.html' # Asegúrate que esta plantilla exista

    def get(self, request, conversation_id):
        conversation = get_object_or_404(Conversation, id=conversation_id)
        if request.session.get('chatbot_conversation_id') != str(conversation_id):
            request.session['chatbot_conversation_id'] = str(conversation_id)
        messages = conversation.messages.all().order_by('timestamp')
        form = MessageForm()
        user_identifier = request.session.get('chatbot_user_nickname', f"Usuario_{str(conversation_id)[:8]}")
        context = {
            'conversation': conversation, # Pasa el objeto Conversation para el action del form
            'messages': messages,
            'form': form,
            'user_identifier': user_identifier,
        }
        return render(request, self.template_name, context)

    def post(self, request, conversation_id):
        conversation = get_object_or_404(Conversation, id=conversation_id)
        form = MessageForm(request.POST, request.FILES) 
        
        user_identifier = request.session.get('chatbot_user_nickname', f"Usuario_{str(conversation_id)[:8]}")
        
        if form.is_valid():
            user_input_text = form.cleaned_data.get('user_input')
            uploaded_image_file = form.cleaned_data.get('image_upload')

            user_message_obj = Message(conversation=conversation, content=user_input_text, is_bot=False)
            if uploaded_image_file:
                user_message_obj.image = uploaded_image_file
            
            input_for_llm = user_input_text if user_input_text else ""

            if uploaded_image_file:
                print(f"--- VIEW DEBUG: Imagen '{uploaded_image_file.name}' recibida para CNN. ---")
                predicted_desease_obj, confidence_percent = cnn_image_processor.predict_from_image_file(uploaded_image_file)
                
                if predicted_desease_obj:
                    print(f"--- VIEW DEBUG: CNN predijo: {predicted_desease_obj.name_desease} ({confidence_percent:.2f}%) ---")
                    user_message_obj.cnn_predicted_desease = predicted_desease_obj
                    user_message_obj.cnn_confidence = confidence_percent 
                    
                    cnn_info_for_llm = (
                        f"El usuario ha subido una imagen. "
                        f"Un análisis por un sistema de IA visual sugiere que podría ser "
                        f"'{predicted_desease_obj.name_desease}' con una confianza del {confidence_percent:.1f}%. "
                    )
                    if input_for_llm:
                        input_for_llm = f"{cnn_info_for_llm} Además, el usuario comentó: '{input_for_llm}'"
                    else:
                        input_for_llm = cnn_info_for_llm + "Por favor, considera esta información visual en tu orientación."
                else:
                    print("--- VIEW DEBUG: CNN no devolvió una predicción de enfermedad válida. ---")
                    cnn_info_for_llm = "El usuario ha subido una imagen, pero el análisis visual no arrojó un resultado específico. "
                    if input_for_llm:
                        input_for_llm = f"{cnn_info_for_llm} El comentario del usuario fue: '{input_for_llm}'"
                    else:
                        input_for_llm = cnn_info_for_llm + "Procede basándote en el texto si lo hay, o pregunta por más detalles de la imagen."
            
            user_message_obj.save() # Guardar el mensaje del usuario (con o sin info de CNN)
            
            if not input_for_llm and not uploaded_image_file:
                input_for_llm ="El usuario envió un mensaje vacío." # El form.clean() debería prevenir esto

            print(f"--- VIEW DEBUG: Input final para LLM DermaBot: '{input_for_llm}' ---")
            bot_response_text = derma_agent_llm.get_response(
                user_input=input_for_llm,
                conversation_id=str(conversation.id),
                user_identifier=user_identifier
            )
            Message.objects.create(conversation=conversation, content=bot_response_text, is_bot=True)
            
            return redirect('chatbot:chat_window', conversation_id=conversation.id)
        
        messages = conversation.messages.all().order_by('timestamp')
        context = {
            'conversation': conversation,
            'messages': messages,
            'form': form, 
            'user_identifier': user_identifier,
        }
        return render(request, self.template_name, context)

class StartNewChatSessionView(View):
    def get(self, request):
        new_conversation = Conversation.objects.create()
        request.session['chatbot_conversation_id'] = str(new_conversation.id)
        return redirect('chatbot:chat_window', conversation_id=new_conversation.id)