# chatbot/services/openai_agent_service.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver 
# from langgraph.checkpoint.sqlite import SqliteSaver # Alternativa para persistencia
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Importa el modelo Desease unificado desde los modelos de ESTA MISMA APP ('chatbot')
from ..models import Desease as KnownDesease # Usamos un alias para mantener consistencia interna si se prefiere

def get_deseases_prompt_text():
    deseases = KnownDesease.objects.all()
    if not deseases:
        print("--- DEBUG AGENT: ADVERTENCIA - No se encontraron enfermedades en la BD para get_deseases_prompt_text().")
        return " (Lista de enfermedades de referencia no disponible. Por favor, configúrala en el panel de administración.)"
    
    text = "\nLista de Referencia de Afecciones Cutáneas (Descripciones MUY CORTAS):\n"
    for d in deseases:
        # Asegurarse de usar los nombres de campo correctos de tu modelo Desease
        text += f"- {d.name_desease}: {d.short_description_for_llm}\n"
    return text

class DermaBotAgent:
    _instance = None

    def __init__(self):
        print("--- DEBUG AGENT: Iniciando __init__ de DermaBotAgent ---")
        if not os.getenv("OPENAI_API_KEY"):
            print("--- DEBUG AGENT: ERROR FATAL - OPENAI_API_KEY no está configurada.")
            raise ValueError("OPENAI_API_KEY no está configurada.")

        # Puedes ajustar max_tokens si las respuestas con info de CNN necesitan más espacio
        self.max_output_tokens = 300 
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, max_tokens=self.max_output_tokens) 
        self.checkpointer = MemorySaver()

        self.base_system_prompt_content = """Eres DermaBot, un asistente virtual en español especializado en orientación dermatológica preliminar.
Tu nombre es DermaBot. El usuario es '{{user_identifier}}', conversación ID '{{conversation_id_thread}}'.
NO DIAGNOSTICAS. Tu objetivo es guiar y educar MUY generalmente. No des tratamientos.
{deseases_info_placeholder} <!-- Aquí se insertará la lista de enfermedades -->

**PROTOCOLO DE INTERACCIÓN ESTRICTO:**
1.  **SALUDO INICIAL:** Si es el primer mensaje del bot en esta conversación, saluda: "¡Hola! Soy DermaBot. ¿Podrías describirme tu problema de piel y dónde se localiza, o subir una imagen si lo prefieres?"
2.  **MANEJO DE INFORMACIÓN DE IMAGEN (CNN):**
    *   Si el mensaje del usuario indica que se ha subido una imagen y se proporciona una sugerencia de un análisis visual previo (CNN) (ej: "Análisis de imagen sugiere: Acné con 75% de confianza"), **ACUSA RECIBO** de esta información ANTES de hacer tu pregunta.
    *   Ejemplo de acuse de recibo: "Entendido, gracias por la imagen. El análisis preliminar sugiere que podría ser [Enfermedad de CNN]."
    *   Luego, **continúa con UNA pregunta de la "LISTA DE PREGUNTAS GENERALES"** para obtener más contexto del usuario, considerando la sugerencia de la CNN. No asumas que la sugerencia de la CNN es definitiva; úsala para guiar tus preguntas.
3.  **UNA PREGUNTA A LA VEZ:** Ya sea después del saludo o después de acusar recibo de la info de la CNN, formula **SOLAMENTE UNA** pregunta de la "LISTA DE PREGUNTAS GENERALES". Elige la pregunta más relevante que aún no haya sido respondida o aclarada. Intenta seguir el orden de la lista.
4.  **ADAPTACIÓN INTELIGENTE:** Omite preguntas de la lista general si la información ya fue dada (por texto o por la sugerencia de la CNN si es muy clara).
5.  **ORIENTACIÓN Y ADVERTENCIA FINAL:** Cuando consideres que tienes suficiente información (idealmente después de que el usuario haya respondido a 2-4 preguntas clave), ofrece una orientación TENTATIVA sobre 1 o MÁXIMO 2 posibles afecciones de tu lista de referencia. Explica brevemente por qué. INMEDIATAMENTE DESPUÉS, concluye con la "Advertencia Médica Obligatoria" completa. Tu turno termina después de la advertencia; no hagas más preguntas.
6.  **RESPUESTAS CONCISAS:** Mantén tus preguntas y respuestas directas y fáciles de entender.

**LISTA DE PREGUNTAS GENERALES (Formula UNA por turno, en orden de relevancia y si no ha sido respondida):**
    P1. ¿Desde cuándo tienes esta afección o estos síntomas? ¿Aparecieron de repente o de forma gradual?
    P2. ¿Cómo describirías exactamente la apariencia de la lesión o la piel afectada? (Por ejemplo: ¿Es una mancha, un grano, una ampolla, piel seca, escamosa, enrojecida, con pus, etc.? ¿Qué color y forma tiene?)
    P3. ¿Sientes alguna molestia como picazón (prurito), dolor, ardor, quemazón, tirantez o sensibilidad al tacto en la zona? Si es así, ¿qué tan intensa es?
    P4. ¿Has notado si hay algo que parezca empeorar los síntomas (ej: sol, calor, frío, estrés, ciertos alimentos, contacto con algún producto, ropa específica)?
    P5. ¿Hay algo que parezca mejorar los síntomas (ej: alguna crema, frío, descanso)?
    P6. ¿Has tenido síntomas similares antes? Si es así, ¿recibiste algún diagnóstico o tratamiento?
    P7. ¿Tienes algún otro síntomas general, aunque no parezca relacionado, como fiebre, cansancio, dolor articular, pérdida de peso o malestar general?
    P8. ¿Estás tomando alguna medicación actualmente (incluyendo suplementos o remedios herbales)? ¿Tienes alguna alergia conocida a medicamentos o a otras sustancias?
    P9. ¿Alguien más en tu familia o entorno cercano tiene problemas de piel similares o alguna enfermedad cutánea importante?

**ADVERTENCIA MÉDICA OBLIGATORIA (Al finalizar la orientación):**
"Recuerda, esta es solo una orientación general basada en la información que me has dado y NO reemplaza una consulta médica profesional. Es fundamental que visites a un dermatólogo para una evaluación adecuada, un diagnóstico preciso y el tratamiento correcto. Por favor, no te automediques ni demores la consulta con un especialista."

Si el usuario pide diagnóstico/tratamiento, reitera amablemente tus limitaciones y la necesidad de ver a un médico.
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
        print(f"--- DEBUG AGENT: call_model_node - User Identifier: {user_identifier}, Thread ID: {thread_id}")

        current_deseases_info = get_deseases_prompt_text()
        
        system_content_with_deseases = self.base_system_prompt_content.replace(
            "{deseases_info_placeholder}", current_deseases_info
        )
        
        print(f"--- DEBUG AGENT: call_model_node - System Prompt preparado. Mensajes actuales en estado: {state['messages']}")

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
            print(f"--- DEBUG AGENT: call_model_node - Respuesta CRUDA del LLM: {response}")
            return {"messages": [response]} 
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR al invocar el LLM en call_model_node: {e} !!!!!!!!!")
            error_ai_message = AIMessage(content="Hubo un problema técnico al procesar tu consulta con el asistente. Por favor, intenta de nuevo más tarde.")
            return {"messages": [error_ai_message]}


    def get_response(self, user_input: str, conversation_id: str, user_identifier: str = "Usuario"):
        print(f"\n--- DEBUG AGENT: Entrando a get_response ---")
        print(f"--- DEBUG AGENT: get_response - User Input: '{user_input}', Conversation ID: '{conversation_id}', User Identifier: '{user_identifier}'")
        
        langgraph_config = {
            "configurable": {
                "thread_id": conversation_id,
                "user_name": user_identifier,
            }
        }
        current_input_message = HumanMessage(content=user_input) # El input para el LLM, que puede incluir info de la CNN
        print(f"--- DEBUG AGENT: get_response - Mensaje actual del usuario (input_for_llm) a enviar a LangGraph: {current_input_message}")

        try:
            response_state = self.graph_app.invoke(
                {"messages": [current_input_message]},
                config=langgraph_config
            )
        except Exception as e:
            print(f"!!!!!!!! DEBUG AGENT: ERROR durante graph_app.invoke en get_response: {e} !!!!!!!!!")
            return "Lo siento, ocurrió un error crítico al procesar tu solicitud."

        print(f"--- DEBUG AGENT: get_response - Estado de respuesta completo de LangGraph: {response_state}")

        if response_state and response_state.get("messages"):
            all_messages_in_state = response_state.get("messages", [])
            last_message = all_messages_in_state[-1] if all_messages_in_state else None
            print(f"--- DEBUG AGENT: get_response - Último mensaje obtenido del estado: {last_message}")

            if not last_message or not hasattr(last_message, 'content'):
                 print("--- DEBUG AGENT: get_response - No se obtuvo un último mensaje válido (None o sin .content).")
                 return "Lo siento, no pude generar una respuesta inteligible en este momento."

            if "Hubo un problema técnico" in last_message.content:
                print("--- DEBUG AGENT: get_response - Devolviendo mensaje de error técnico propagado.")
                return last_message.content
            
            # El saludo inicial ahora está explícitamente en el prompt
            # y la lógica de cómo el bot saluda (solo una vez) debe ser manejada por el LLM
            # basándose en el historial de la conversación que recibe.
            
            print(f"--- DEBUG AGENT: get_response - Devolviendo contenido del último mensaje: '{last_message.content}'")
            return last_message.content
        
        print("--- DEBUG AGENT: get_response - response_state o response_state.get('messages') fue None o lista vacía.")
        return "Lo siento, no pude procesar tu mensaje. Intenta de nuevo."

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("--- DEBUG AGENT: Creando NUEVA instancia de DermaBotAgent (Singleton) ---")
            cls._instance = cls()
        return cls._instance