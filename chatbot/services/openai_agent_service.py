# chatbot/services/openai_agent_service.py
import os
import uuid # Para la conversión de conversation_id a UUID en get_response
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Importa los modelos necesarios de la app chatbot
from ..models import Desease as KnownDesease, Conversation, MedicalSummary 

def get_deseases_prompt_text():
    deseases = KnownDesease.objects.all()
    if not deseases:
        print("--- DEBUG AGENT: ADVERTENCIA - No se encontraron enfermedades en la BD para get_deseases_prompt_text().")
        return " (Lista de enfermedades de referencia no disponible. Configurar en admin.)"
    
    text = "\nLista de Referencia de Afecciones Cutáneas (Descripciones MUY CORTAS para IA):\n"
    for d in deseases:
        text += f"- {d.name_desease}: {d.short_description_for_llm}\n"
    return text

class DermaBotAgent:
    _instance = None

    def __init__(self):
        print("--- DEBUG AGENT: Iniciando __init__ de DermaBotAgent ---")
        if not os.getenv("OPENAI_API_KEY"):
            print("--- DEBUG AGENT: ERROR FATAL - OPENAI_API_KEY no está configurada.")
            raise ValueError("OPENAI_API_KEY no está configurada.")

        self.max_output_tokens = 450 # Aumentado para dar espacio al resumen + orientación
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, max_tokens=self.max_output_tokens) 
        self.checkpointer = MemorySaver()

        self.base_system_prompt_content = """Eres DermaBot, un asistente virtual en español para orientación dermatológica preliminar.
Tu nombre es DermaBot. El usuario es '{{user_identifier}}', conversación ID '{{conversation_id_thread}}'.
NO DIAGNOSTICAS. Tu objetivo es guiar y educar MUY generalmente. No des tratamientos.
{deseases_info_placeholder} <!-- Aquí se insertará la lista de enfermedades -->

**PROTOCOLO DE INTERACCIÓN ESTRICTO:**
1.  **SALUDO INICIAL:** Si es el primer mensaje del bot, saluda: "¡Hola! Soy DermaBot. ¿Podrías describirme tu problema de piel y dónde se localiza, o subir una imagen si lo prefieres?"
2.  **MANEJO DE INFORMACIÓN DE IMAGEN (CNN):**
    *   Si el mensaje del usuario indica que se ha subido una imagen y se proporciona una sugerencia de un análisis visual previo (CNN) (ej: "Análisis de imagen sugiere: Acné con 75% de confianza"), **ACUSA RECIBO** de esta información. Ejemplo: "Entendido, gracias por la imagen. El análisis visual sugiere que podría ser [Enfermedad de CNN]."
    *   Luego, **continúa con UNA pregunta de la "LISTA DE PREGUNTAS GENERALES"** para obtener más contexto.
3.  **UNA PREGUNTA A LA VEZ:** Formula SOLAMENTE UNA pregunta de la "LISTA DE PREGUNTAS GENERALES". Elige la más relevante no respondida.
4.  **ADAPTACIÓN INTELIGENTE:** Omite preguntas si la información ya fue dada.
5.  **GENERACIÓN DE RESUMEN, ORIENTACIÓN Y ADVERTENCIA FINAL:** 
    *   Cuando tengas suficiente información (2-4 respuestas clave), **en tu respuesta final al usuario**, primero incluye un bloque de resumen ESTRUCTURADO y OCULTO para procesamiento interno. Este bloque debe empezar con `###INICIO_RESUMEN_MEDICO###` y terminar con `###FIN_RESUMEN_MEDICO###`.
    *   **Formato del Resumen (entre las etiquetas):**
        Motivo Principal: [Descripción concisa del problema principal del usuario]
        Síntomas Reportados: [Lista de síntomas clave como: "manchas rojas", "picazón intensa", "ampollas pequeñas", etc.]
        Localización: [Partes del cuerpo afectadas, ej: "antebrazo derecho", "codos y rodillas"]
        Duración: [Tiempo desde el inicio de los síntomas, ej: "dos semanas", "varios años de forma intermitente"]
        Factores Agravantes: [Qué empeora los síntomas, si se mencionó, ej: "estrés", "chocolate"]
        Factores de Alivio: [Qué mejora los síntomas, si se mencionó, ej: "cremas hidratantes", "sol moderado"]
        Antecedentes Relevantes: [Cualquier antecedente médico o familiar, o diagnóstico previo, ej: "padre con psoriasis", "diagnóstico tentativo previo de psoriasis"]
        Análisis de Imagen (CNN): [Si se proporcionó, resume aquí lo que la CNN sugirió. Ej: "CNN sugiere: Melanoma (Confianza: 85%)" o "CNN: Sin resultado específico para la imagen."]
    *   **DESPUÉS de las etiquetas y el resumen oculto, en la MISMA respuesta al usuario**, ofrece una orientación TENTATIVA sobre 1 o MÁXIMO 2 posibles afecciones de tu lista de referencia. Explica brevemente por qué.
    *   INMEDIATAMENTE DESPUÉS de la orientación, concluye con la "Advertencia Médica Obligatoria" completa. Tu turno termina aquí.
6.  **RESPUESTAS CONCISAS.**

**LISTA DE PREGUNTAS GENERALES (Haz UNA por turno):**
    P1. ¿Desde cuándo tienes esta afección o estos síntomas? ¿Aparición súbita o gradual?
    P2. ¿Cómo es la apariencia de la lesión/piel? (Color, forma, textura: manchas, granos, ampollas, etc.)
    P3. ¿Sientes picazón, dolor, ardor, calor, o tirantez? ¿Intensidad?
    P4. ¿Algo empeora los síntomas (sol, estrés, alimentos, productos)?
    P5. ¿Algo los mejora (cremas, frío)?
    P6. ¿Síntomas similares antes? ¿Diagnóstico/tratamiento previo?
    P7. ¿Otros síntomas generales (fiebre, cansancio, dolor articular)?
    P8. ¿Medicación actual (suplementos, hierbas)? ¿Alergias conocidas?
    P9. ¿Antecedentes familiares de problemas de piel?

**ADVERTENCIA MÉDICA OBLIGATORIA (Al finalizar la orientación):**
"Recuerda, esta es solo una orientación general basada en la información que me has dado y NO reemplaza una consulta médica profesional. Es fundamental que visites a un dermatólogo para una evaluación adecuada, un diagnóstico preciso y el tratamiento correcto. Por favor, no te automediques ni demores la consulta con un especialista."

Si pide diagnóstico/tratamiento, reitera limitaciones.
"""
        # LangGraph Workflow
        workflow = StateGraph(MessagesState)
        workflow.add_node("model", self.call_model_node)
        workflow.add_edge(START, "model")
        self.graph_app = workflow.compile(checkpointer=self.checkpointer)
        print(f"--- DEBUG AGENT: DermaBotAgent (OpenAI) inicializado. max_tokens={self.max_output_tokens}. Checkpointer: {type(self.checkpointer).__name__} ---")

    def call_model_node(self, state: MessagesState, config: dict):
        print("\n--- DEBUG AGENT: Entrando a call_model_node ---")
        cfg_configurable = config.get("configurable", {})
        user_identifier = cfg_configurable.get("user_name", "Usuario")
        thread_id = cfg_configurable.get("thread_id", "default_thread")
        # print(f"--- DEBUG AGENT: call_model_node - User Identifier: {user_identifier}, Thread ID: {thread_id}")

        current_deseases_info = get_deseases_prompt_text()
        
        system_content_with_deseases = self.base_system_prompt_content.replace(
            "{deseases_info_placeholder}", current_deseases_info
        )
        
        print(f"--- DEBUG AGENT: call_model_node - Historial para LLM: {state['messages']}")

        current_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_content_with_deseases),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        chain = current_prompt | self.model
        
        try:
            response = chain.invoke({
                "messages": state["messages"], 
                "user_identifier": user_identifier,
                "conversation_id_thread": thread_id,
            })
            print(f"--- DEBUG AGENT: call_model_node - Respuesta CRUDA del LLM: {response.content}") # Imprimir response.content
            return {"messages": [response]} 
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR al invocar el LLM en call_model_node: {e} !!!!!!!!!")
            error_ai_message = AIMessage(content="Hubo un problema técnico al procesar tu consulta con el asistente. Por favor, intenta de nuevo más tarde.")
            return {"messages": [error_ai_message]}


    def _extract_and_save_medical_summary(self, llm_response_content: str, conversation_obj: Conversation):
        """
        Extrae el resumen médico si está presente en la respuesta del LLM y lo guarda.
        Devuelve la parte de la respuesta que es para el usuario.
        """
        start_tag = "###INICIO_RESUMEN_MEDICO###"
        end_tag = "###FIN_RESUMEN_MEDICO###"
        
        user_facing_response = llm_response_content # Por defecto, toda la respuesta es para el usuario
        raw_summary_text = ""

        start_index = llm_response_content.find(start_tag)
        end_index = llm_response_content.find(end_tag)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extraer el resumen crudo
            raw_summary_text = llm_response_content[start_index + len(start_tag) : end_index].strip()
            
            # La respuesta para el usuario es lo que está ANTES del resumen y/o DESPUÉS del resumen
            # Si el resumen está al principio:
            if start_index == 0:
                 user_facing_response = llm_response_content[end_index + len(end_tag) :].strip()
            # Si el resumen está al final (menos probable con el prompt actual):
            elif end_index + len(end_tag) == len(llm_response_content):
                 user_facing_response = llm_response_content[:start_index].strip()
            # Si está en el medio (aún menos probable, pero por si acaso):
            else:
                 user_facing_response_before = llm_response_content[:start_index].strip()
                 user_facing_response_after = llm_response_content[end_index + len(end_tag) :].strip()
                 user_facing_response = (user_facing_response_before + " " + user_facing_response_after).strip()

            print(f"--- DEBUG AGENT: Resumen Médico Crudo Extraído:\n{raw_summary_text}\n---------------------------------")
            print(f"--- DEBUG AGENT: Respuesta para el Usuario (post-extracción resumen):\n{user_facing_response}\n---------------------------------")

            # Guardar o actualizar el MedicalSummary
            summary_data = {"summary_text_generated_by_llm": raw_summary_text}
            # Parseo simple de campos específicos
            lines = raw_summary_text.split('\n')
            orientation_text_parts = []
            in_orientation_section = True # Asumimos que el resumen está antes de la orientación directa al usuario

            for line in lines:
                if ":" in line:
                    key_raw, value = line.split(":", 1)
                    key = key_raw.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") # Normalizar clave
                    value = value.strip()
                    if key == "motivo_principal": summary_data["main_complaint"] = value
                    elif key == "síntomas_reportados": summary_data["symptoms_reported"] = value
                    elif key == "localización": summary_data["location_of_symptoms"] = value
                    elif key == "duración": summary_data["duration_of_symptoms"] = value
                    elif key == "factores_agravantes": summary_data["aggravating_factors"] = value
                    elif key == "factores_de_alivio": summary_data["alleviating_factors"] = value
                    elif key == "antecedentes_relevantes": summary_data["previous_history"] = value
                    elif key == "análisis_de_imagen_cnn": summary_data["image_analysis_summary_from_cnn"] = value
            
            # La orientación tentativa es la parte que el LLM dirige al usuario
            # Esta podría estar en `user_facing_response` si el LLM sigue el prompt de ponerlo después del resumen.
            summary_data["tentative_orientation_by_llm"] = user_facing_response 
            
            summary_obj, created = MedicalSummary.objects.update_or_create(
                conversation=conversation_obj,
                defaults=summary_data
            )
            if created:
                print(f"--- DEBUG AGENT: Creado MedicalSummary para conversación {conversation_obj.id}")
            else:
                print(f"--- DEBUG AGENT: Actualizado MedicalSummary para conversación {conversation_obj.id}")
        else:
            print("--- DEBUG AGENT: No se encontraron etiquetas de resumen médico completas en la respuesta del LLM.")
            # Si no hay bloque de resumen, y la respuesta parece la orientación final (contiene la advertencia)
            # la guardamos como la orientación.
            advertencia_inicio = "Recuerda, esta es solo una orientación general"
            if advertencia_inicio.lower() in user_facing_response.lower():
                summary_obj, created = MedicalSummary.objects.get_or_create(conversation=conversation_obj)
                if not summary_obj.tentative_orientation_by_llm: # Solo si no hay ya una orientación guardada
                    summary_obj.tentative_orientation_by_llm = user_facing_response
                    summary_obj.save()
                    print(f"--- DEBUG AGENT: Guardada orientación final en MedicalSummary para {conversation_obj.id}")

        return user_facing_response.strip()


    def get_response(self, user_input: str, conversation_id: str, user_identifier: str = "Usuario"):
        print(f"\n--- DEBUG AGENT: Entrando a get_response ---")
        print(f"--- DEBUG AGENT: get_response - User Input: '{user_input}', Conv ID: '{conversation_id}', User: '{user_identifier}'")
        
        langgraph_config = { "configurable": { "thread_id": conversation_id, "user_name": user_identifier, } }
        current_input_message = HumanMessage(content=user_input)
        
        try:
            response_state = self.graph_app.invoke({"messages": [current_input_message]}, config=langgraph_config)
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR durante graph_app.invoke: {e} !!!!!!!!!")
            return "Lo siento, ocurrió un error catastrófico al procesar su solicitud."

        print(f"--- DEBUG AGENT: get_response - Estado de respuesta completo de LangGraph: {response_state}")

        llm_full_response_content = ""
        if response_state and response_state.get("messages"):
            all_messages_in_state = response_state.get("messages", [])
            last_message_obj = all_messages_in_state[-1] if all_messages_in_state else None
            print(f"--- DEBUG AGENT: get_response - Último mensaje obj obtenido: {last_message_obj}")

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

        # --- Lógica para extraer resumen y devolver respuesta al usuario ---
        try:
            # Convertir conversation_id (string) a UUID para la consulta
            conversation_uuid = uuid.UUID(conversation_id)
            conversation_obj = Conversation.objects.get(id=conversation_uuid)
            user_facing_content = self._extract_and_save_medical_summary(llm_full_response_content, conversation_obj)
            
            # Si user_facing_content queda vacío después de extraer el resumen, es porque
            # toda la respuesta del LLM era el bloque de resumen (lo cual no debería pasar según el prompt).
            # En ese caso, devolvemos un mensaje genérico o la respuesta completa original.
            if not user_facing_content.strip() and llm_full_response_content:
                print("--- DEBUG AGENT: user_facing_content vacío post-extracción, devolviendo llm_full_response_content (sin resumen explícito para el usuario).")
                # Esto podría pasar si el LLM solo da el resumen oculto y no la parte para el usuario.
                # Es una señal de que el prompt necesita ajuste o el LLM no lo siguió.
                # Devolvemos la respuesta completa para no perderla.
                return llm_full_response_content

            print(f"--- DEBUG AGENT: get_response - Devolviendo contenido final para el usuario: '{user_facing_content}'")
            return user_facing_content
        except Conversation.DoesNotExist:
            print(f"--- DEBUG AGENT: ERROR CRÍTICO - No se encontró Conversation con ID {conversation_id} para guardar resumen.")
            return llm_full_response_content 
        except Exception as e:
            print(f"--- DEBUG AGENT: ERROR al procesar/guardar resumen médico: {e}")
            return llm_full_response_content

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("--- DEBUG AGENT: Creando NUEVA instancia de DermaBotAgent (Singleton) ---")
            cls._instance = cls()
        return cls._instance