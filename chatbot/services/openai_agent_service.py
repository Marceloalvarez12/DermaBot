# Detection/services/openai_dermabot_agent.py 
# (o chatbot/services/openai_agent_service.py)

import os
import uuid 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver # Usando MemorySaver como en tu funcional
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Ajusta la ruta de importación de modelos según la ubicación de este archivo
# Si este archivo está en Detection/services/ y los modelos en Detection/models.py:
from ..models import Desease as KnownDesease, Conversation, MedicalSummary 
# Si este archivo estuviera en chatbot/services/ y modelos en chatbot/models.py:
# from ..models import Desease as KnownDesease, Conversation, MedicalSummary

# LANGGRAPH_SQLITE_DB_PATH ya no es necesario con MemorySaver

def get_deseases_prompt_text():
    deseases = KnownDesease.objects.all() # KnownDesease es un alias para Desease
    if not deseases:
        print("--- DEBUG AGENT: ADVERTENCIA - No se encontraron enfermedades en la BD para get_deseases_prompt_text().")
        return " (Lista de enfermedades de referencia no disponible. Configurar en admin.)"
    
    text = "\nLista de Referencia de Afecciones Cutáneas (Descripciones MUY CORTAS para IA):\n"
    for d in deseases:
        name = d.name_desease if d.name_desease else "Nombre no disponible"
        # Asegúrate de que el campo short_description_for_llm exista en tu modelo Desease
        desc = d.short_description_for_llm if hasattr(d, 'short_description_for_llm') and d.short_description_for_llm else "Descripción breve no disponible."
        text += f"- {name}: {desc}\n"
    return text

class DermaBotAgent:
    _instance = None

    def __init__(self):
        print("--- DEBUG AGENT: Iniciando __init__ de DermaBotAgent ---")
        
        openai_api_key_val = None
        # Intenta leer desde settings.py primero (que debería leer desde .env)
        try:
            from django.conf import settings as django_settings
            openai_api_key_val = getattr(django_settings, 'OPENAI_API_KEY', None)
        except ImportError:
            print("--- DEBUG AGENT: Django settings no disponibles, intentando os.getenv ---")
            pass # Django no está disponible o no está configurado (ej. pruebas unitarias puras del agente)

        if not openai_api_key_val: # Si no se encontró en settings, intenta os.getenv
            openai_api_key_val = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key_val:
            print("--- DEBUG AGENT: ERROR FATAL - OPENAI_API_KEY no está configurada ni en settings.py ni como variable de entorno.")
            raise ValueError("OPENAI_API_KEY no está configurada.")

        self.max_output_tokens = 450 # Espacio para resumen + orientación + posible respuesta general
        self.model = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.6, 
            max_tokens=self.max_output_tokens,
            api_key=openai_api_key_val
        )
        
        self.checkpointer = MemorySaver() # Usando MemorySaver como en tu proyecto funcional

        self.base_system_prompt_content = """Eres DermaBot, un asistente virtual en español para orientación dermatológica preliminar y para responder preguntas generales sobre dermatología.
Tu nombre es DermaBot. El usuario es '{user_identifier}', conversación ID '{conversation_id_thread}'.
NO DIAGNOSTICAS. Tu objetivo es guiar, educar MUY generalmente y responder preguntas informativas. No des tratamientos.

{deseases_info_placeholder} <!-- Lista de afecciones de referencia para orientación específica -->

**MODOS DE RESPUESTA:**
1.  **ORIENTACIÓN ESPECÍFICA (Protocolo Estricto):** Si el usuario describe un problema de piel personal, sigue el "PROTOCOLO DE INTERACCIÓN ESTRICTO" de abajo para recolectar síntomas y ofrecer una orientación TENTATIVA basada en la "{deseases_info_placeholder}".
2.  **PREGUNTAS GENERALES SOBRE DERMATOLOGÍA:** Si el usuario hace una pregunta general sobre dermatología (ej: "¿Qué es el acné?", "¿Cómo funciona el protector solar?", "¿Cuáles son los tipos de piel?"), que no sea sobre un problema personal o que no requiera seguir el protocolo de preguntas, puedes responder usando tu conocimiento general. 
    *   Mantén tus respuestas informativas, claras, concisas y en un lenguaje accesible.
    *   Cita fuentes si es posible y apropiado (aunque como LLM, esto puede ser simulado o basado en tu entrenamiento).
    *   **IMPORTANTE:** Incluso para preguntas generales, si la respuesta podría interpretarse como un consejo médico específico para una condición, SIEMPRE incluye una versión de la "Advertencia Médica Obligatoria" adaptada. Ejemplo: "Esta es información general y no reemplaza el consejo de un dermatólogo."

**PROTOCOLO DE INTERACCIÓN ESTRICTO (Para orientación sobre problemas de piel del usuario):**
1.  **SALUDO INICIAL:** Si es el primer mensaje del bot y el usuario describe un problema, saluda: "¡Hola! Soy DermaBot. Entendido. Para ayudarte mejor con tu problema de [lo que describió el usuario], ¿podrías..." y haz la primera pregunta relevante. Si solo saluda, pregunta: "¿Podrías describirme tu problema de piel y dónde se localiza, o subir una imagen si lo prefieres?"
2.  **MANEJO DE INFORMACIÓN DE IMAGEN (CNN):** Si el mensaje del usuario indica que se ha subido una imagen y se proporciona una sugerencia de un análisis visual previo (CNN) (ej: "Contexto de imagen: El análisis preliminar..."), **ACUSA RECIBO** de esta información. Ejemplo: "Entendido, gracias por la imagen. El análisis visual sugiere que podría ser [Enfermedad de CNN]." Luego, **continúa con UNA pregunta de la "LISTA DE PREGUNTAS GENERALES"** para obtener más contexto.
3.  **UNA PREGUNTA A LA VEZ:** Formula SOLAMENTE UNA pregunta de la "LISTA DE PREGUNTAS GENERALES". Elige la más relevante no respondida. Intenta seguir el orden de la lista si es lógico.
4.  **ADAPTACIÓN INTELIGENTE:** Omite preguntas si la información ya fue dada por el usuario o por el análisis de imagen. No repitas.
5.  **GENERACIÓN DE RESUMEN, ORIENTACIÓN Y ADVERTENCIA FINAL:** 
    *   Cuando tengas suficiente información (usualmente después de 2-4 respuestas clave del usuario), **SOLO SI ESTÁS SIGUIENDO ESTE PROTOCOLO ESTRICTO para un problema de piel personal,** en tu respuesta final al usuario, primero incluye un bloque de resumen ESTRUCTURADO y OCULTO para procesamiento interno. Este bloque debe empezar con `###INICIO_RESUMEN_MEDICO###` y terminar con `###FIN_RESUMEN_MEDICO###`.
    *   **Formato del Resumen (entre las etiquetas):**
        Motivo Principal: [Descripción concisa del problema principal del usuario]
        Síntomas Reportados: [Lista de síntomas clave como: "manchas rojas", "picazón intensa", "ampollas pequeñas", etc.]
        Localización: [Partes del cuerpo afectadas, ej: "antebrazo derecho", "codos y rodillas"]
        Duración: [Tiempo desde el inicio de los síntomas, ej: "dos semanas", "varios años de forma intermitente"]
        Factores Agravantes: [Qué empeora los síntomas, si se mencionó, ej: "estrés", "chocolate"]
        Factores de Alivio: [Qué mejora los síntomas, si se mencionó, ej: "cremas hidratantes", "sol moderado"]
        Antecedentes Relevantes: [Cualquier antecedente médico o familiar, o diagnóstico previo, ej: "padre con psoriasis", "diagnóstico tentativo previo de psoriasis"]
        Análisis de Imagen (CNN): [Si se proporcionó, resume aquí lo que la CNN sugirió. Ej: "CNN sugiere: Melanoma (Confianza: 85%)" o "CNN: Sin resultado específico para la imagen."]
    *   **DESPUÉS de las etiquetas y el resumen oculto (si lo generaste), en la MISMA respuesta al usuario**, ofrece una orientación TENTATIVA sobre 1 o MÁXIMO 2 posibles afecciones de tu lista de referencia {deseases_info_placeholder}. Explica brevemente por qué.
    *   INMEDIATAMENTE DESPUÉS de la orientación, concluye con la "Advertencia Médica ObligatorIA" completa. Tu turno termina aquí; no hagas más preguntas después de la advertencia.
6.  **RESPUESTAS CONCISAS.**

**LISTA DE PREGUNTAS GENERALES (Para el protocolo estricto, haz UNA por turno si no ha sido respondida):**
    P1. ¿Desde cuándo tienes esta afección o estos síntomas? ¿Aparecieron de repente o de forma gradual?
    P2. ¿Cómo describirías exactamente la apariencia de la lesión o la piel afectada? (Por ejemplo: ¿Es una mancha, un grano, una ampolla, piel seca, escamosa, enrojecida, con pus, etc.? ¿Qué color y forma tiene?) (Adapta si ya hay info de una imagen).
    P3. ¿Sientes alguna molestia como picazón (prurito), dolor, ardor, quemazón, tirantez o sensibilidad al tacto en la zona? Si es así, ¿qué tan intensa es?
    P4. ¿Has notado si hay algo que parezca empeorar los síntomas (ej: sol, calor, frío, estrés, ciertos alimentos, contacto con algún producto, ropa específica)?
    P5. ¿Hay algo que parezca mejorar los síntomas (ej: alguna crema, frío, descanso)?
    P6. ¿Has tenido síntomas similares antes? Si es así, ¿recibiste algún diagnóstico o tratamiento?
    P7. ¿Tienes algún otro síntoma general, aunque no parezca relacionado, como fiebre, cansancio, dolor articular, pérdida de peso o malestar general?
    P8. ¿Estás tomando alguna medicación actualmente (incluyendo suplementos o remedios herbales)? ¿Tienes alguna alergia conocida a medicamentos o a otras sustancias?
    P9. ¿Alguien más en tu familia o entorno cercano tiene problemas de piel similares o alguna enfermedad cutánea importante?

**ADVERTENCIA MÉDICA OBLIGATORIA (Al finalizar la ORIENTACIÓN ESPECÍFICA o si una respuesta general podría ser malinterpretada como consejo médico personal):**
"Recuerda, esta es solo una orientación/información general basada en la información que me has dado/solicitado y NO reemplaza una consulta médica profesional. Es fundamental que visites a un dermatólogo para una evaluación adecuada, un diagnóstico preciso y el tratamiento correcto. Por favor, no te automediques ni demores la consulta con un especialista."

Si el usuario pide diagnóstico/tratamiento, reitera limitaciones amablemente.
Si el usuario hace una pregunta muy específica sobre una enfermedad rara o un tratamiento complejo que excede la orientación general, indica que esa pregunta debe ser consultada con un profesional.
"""
        workflow = StateGraph(MessagesState)
        workflow.add_node("model", self.call_model_node)
        workflow.add_edge(START, "model")
        self.graph_app = workflow.compile(checkpointer=self.checkpointer)
        print(f"--- DEBUG AGENT: DermaBotAgent (OpenAI) inicializado. max_tokens={self.max_output_tokens}. Checkpointer: {type(self.checkpointer).__name__} ---")

    def call_model_node(self, state: MessagesState, config: dict):
        print("\n--- DEBUG AGENT: Entrando a call_model_node ---")
        cfg_configurable = config.get("configurable", {})
        user_identifier = cfg_configurable.get("user_name", "Usuario Anónimo") # Default si no se pasa
        thread_id = cfg_configurable.get("thread_id", "default_thread_id_error") # Default para detectar errores
        
        current_deseases_info = get_deseases_prompt_text()
        
        system_prompt_formatted = self.base_system_prompt_content.replace(
            "{user_identifier}", str(user_identifier)
        ).replace(
            "{conversation_id_thread}", str(thread_id)
        ).replace(
            "{deseases_info_placeholder}", current_deseases_info
        )
        
        # Descomentar para depuración extensa del historial
        # print(f"--- DEBUG AGENT: call_model_node - Historial para LLM: {state['messages']}")

        current_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt_formatted),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        chain = current_prompt | self.model
        
        try:
            response = chain.invoke({"messages": state["messages"]}) 
            print(f"--- DEBUG AGENT: call_model_node - Respuesta CRUDA del LLM (primeros 200 chars): {str(response.content)[:200]}...")
            return {"messages": [response]} 
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR al invocar el LLM en call_model_node: {e} !!!!!!!!!")
            error_ai_message = AIMessage(content="Hubo un problema técnico al procesar tu consulta con el asistente. Por favor, intenta de nuevo más tarde.")
            return {"messages": [error_ai_message]}

    def _extract_and_save_medical_summary(self, llm_response_content: str, conversation_obj: Conversation):
        start_tag = "###INICIO_RESUMEN_MEDICO###"
        end_tag = "###FIN_RESUMEN_MEDICO###"
        user_facing_response = llm_response_content
        raw_summary_text = ""
        start_index = llm_response_content.find(start_tag)
        end_index = llm_response_content.find(end_tag)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            raw_summary_text = llm_response_content[start_index + len(start_tag) : end_index].strip()
            
            # La parte para el usuario es lo que viene DESPUÉS del bloque de resumen
            user_facing_response = llm_response_content[end_index + len(end_tag) :].strip()
            # Si no hay nada después (el LLM se detuvo justo después del tag), 
            # o si el bloque de resumen estaba al final de todo y la parte del usuario antes.
            if not user_facing_response.strip():
                user_facing_response = llm_response_content[:start_index].strip()

            print(f"--- DEBUG AGENT: Resumen Médico Crudo Extraído:\n{raw_summary_text}\n---------------------------------")
            print(f"--- DEBUG AGENT: Respuesta para el Usuario (post-extracción resumen):\n{user_facing_response}\n---------------------------------")

            summary_data = {"summary_text_generated_by_llm": raw_summary_text}
            lines = raw_summary_text.split('\n')
            for line in lines:
                if ":" in line:
                    key_raw, value = line.split(":", 1)
                    # Normalizar la clave para que coincida con los nombres de campo del modelo MedicalSummary
                    key = key_raw.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("í", "i").replace("ó", "o").replace("á", "a").replace("é", "e").replace("ú", "u")
                    value = value.strip()
                    
                    # Mapeo explícito a los campos del modelo
                    if key == "motivo_principal": summary_data["main_complaint"] = value
                    elif key == "sintomas_reportados": summary_data["symptoms_reported"] = value # 'í' a 'i'
                    elif key == "localizacion": summary_data["location_of_symptoms"] = value
                    elif key == "duracion": summary_data["duration_of_symptoms"] = value
                    elif key == "factores_agravantes": summary_data["aggravating_factors"] = value
                    elif key == "factores_de_alivio": summary_data["alleviating_factors"] = value
                    elif key == "antecedentes_relevantes": summary_data["previous_history"] = value
                    elif key == "analisis_de_imagen_cnn": summary_data["image_analysis_summary_from_cnn"] = value
            
            # La orientación tentativa es la parte que el LLM dirige al usuario y que NO es el bloque de resumen
            summary_data["tentative_orientation_by_llm"] = user_facing_response.strip()
            
            summary_obj, created = MedicalSummary.objects.update_or_create(
                conversation=conversation_obj,
                defaults=summary_data
            )
            action = "Creado" if created else "Actualizado"
            print(f"--- DEBUG AGENT: {action} MedicalSummary para conversación {conversation_obj.id}")
        else:
            print("--- DEBUG AGENT: No se encontraron etiquetas de resumen médico completas en la respuesta del LLM.")
            advertencia_inicio_lower = "recuerda, esta es solo una orientación general"
            if advertencia_inicio_lower in user_facing_response.lower(): # Comparación insensible a mayúsculas
                summary_obj, created = MedicalSummary.objects.get_or_create(conversation=conversation_obj)
                # Solo guardar si no hay ya una orientación o si es la primera vez que se crea el resumen
                if not summary_obj.tentative_orientation_by_llm or created: 
                    summary_obj.tentative_orientation_by_llm = user_facing_response.strip()
                    summary_obj.save()
                    print(f"--- DEBUG AGENT: Guardada orientación final (sin bloque de resumen explícito) en MedicalSummary para {conversation_obj.id}")

        return user_facing_response.strip() if user_facing_response.strip() else llm_response_content 

    def get_response(self, user_input: str, conversation_id: str, user_identifier: str = "Usuario Anónimo"):
        print(f"\n--- DEBUG AGENT: Entrando a get_response ---")
        print(f"--- DEBUG AGENT: get_response - User Input: '{user_input}', Conv ID: '{conversation_id}', User: '{user_identifier}'")
        
        langgraph_thread_id = str(conversation_id)
        langgraph_config = { "configurable": { "thread_id": langgraph_thread_id, "user_name": str(user_identifier), } }
        current_input_message = HumanMessage(content=user_input)
        
        try:
            response_state = self.graph_app.invoke({"messages": [current_input_message]}, config=langgraph_config)
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR durante graph_app.invoke: {e} !!!!!!!!!")
            return "Lo siento, ocurrió un error catastrófico al procesar su solicitud."

        llm_full_response_content = ""
        if response_state and response_state.get("messages"):
            all_messages_in_state = response_state.get("messages", [])
            last_message_obj = all_messages_in_state[-1] if all_messages_in_state else None

            if not last_message_obj or not hasattr(last_message_obj, 'content'):
                 print("--- DEBUG AGENT: get_response - No se obtuvo un último mensaje válido del LLM.")
                 return "Lo siento, el asistente no pudo generar una respuesta en este momento."
            
            llm_full_response_content = last_message_obj.content
            
            if "Hubo un problema técnico" in llm_full_response_content:
                print("--- DEBUG AGENT: get_response - Devolviendo error técnico propagado.")
                return llm_full_response_content
        else:
            print("--- DEBUG AGENT: get_response - response_state o messages fue None/vacío.")
            return "Lo siento, no se recibió una respuesta estructurada del asistente."

        try:
            conversation_uuid = uuid.UUID(langgraph_thread_id) 
            conversation_obj = Conversation.objects.get(id=conversation_uuid)
            user_facing_content = self._extract_and_save_medical_summary(llm_full_response_content, conversation_obj)
            
            if not user_facing_content.strip() and llm_full_response_content:
                print("--- DEBUG AGENT: user_facing_content vacío post-extracción, devolviendo llm_full_response_content.")
                return llm_full_response_content

            print(f"--- DEBUG AGENT: get_response - Devolviendo contenido final para el usuario: '{user_facing_content}'")
            return user_facing_content
        except Conversation.DoesNotExist:
            print(f"!!!!!!!! DEBUG AGENT: ERROR CRÍTICO - No se encontró Conversation con ID {langgraph_thread_id} para guardar resumen. !!!!!!!!")
            return llm_full_response_content 
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR al procesar/guardar resumen médico: {e} !!!!!!!!")
            return llm_full_response_content

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("--- DEBUG AGENT: Creando NUEVA instancia de DermaBotAgent (Singleton) ---")
            cls._instance = cls()
        return cls._instance