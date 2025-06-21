# chatbot/services/cnn_service.py
import os
import sys
from PIL import Image
from django.conf import settings
# Asegúrate de que las importaciones de Keras/TensorFlow sean consistentes
# Si usas 'tensorflow.keras', sé consistente. Si es solo 'keras', también.
# Dado que en tus errores aparece tensorflow, usaré tensorflow.keras.
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile
from ..models import Desease # Importa desde la app chatbot

class CNNProcessor:
    _instance = None  # Para el Singleton de la clase CNNProcessor
    _model_cnn_internal_instance = None  # Para el modelo Keras (renombrado para claridad)
    
    # Rutas del modelo (ajusta si es necesario)
    _model_path = os.path.join(settings.BASE_DIR, 'chatbot', 'AI_Model', 'skin-cancer-7-classes_MobileNet_ph2_model.h5')
    _weights_path = os.path.join(settings.BASE_DIR, 'chatbot', 'AI_Model', 'skin-cancer-7-classes_MobileNet_ph2_weights.keras')

    def __init__(self):
        """
        Constructor privado. La carga del modelo Keras ocurre aquí si aún no está cargado.
        Este constructor será llamado solo una vez por el método get_instance().
        """
        if CNNProcessor._model_cnn_internal_instance is None:
            print("--- CNN DEBUG: Cargando modelo Keras y pesos por primera vez (dentro de __init__) ---")
            try:
                print(f"--- CNN DEBUG: Intentando cargar modelo desde: {self._model_path} con compile=False ---")
                # Cargar el modelo sin compilar primero
                loaded_model = load_model(
                    self._model_path,
                    custom_objects=None,
                    compile=False  # Crucial para muchos modelos guardados
                )
                print(f"--- CNN DEBUG: Modelo cargado (compile=False). Intentando cargar pesos desde: {self._weights_path} ---")
                loaded_model.load_weights(self._weights_path)
                CNNProcessor._model_cnn_internal_instance = loaded_model # Asignar a la variable de clase
                print("--- CNN DEBUG: Modelo Keras y pesos cargados y asignados exitosamente. ---")
                
                # Opcional: Si necesitas compilarlo después de cargar los pesos para predicción
                # (usualmente no es necesario si solo haces .predict() y no re-entrenamiento)
                # print("--- CNN DEBUG: Compilando modelo (si es necesario)... ---")
                # CNNProcessor._model_cnn_internal_instance.compile(optimizer='adam', loss='categorical_crossentropy') # Ajusta optimizador/loss
                # print("--- CNN DEBUG: Modelo compilado. ---")

            except FileNotFoundError:
                print(f"!!!!!!!! CNN ERROR FATAL: Archivo de modelo o pesos NO ENCONTRADO. !!!!!!!!")
                print(f"Modelo esperado en: {self._model_path}")
                print(f"Pesos esperados en: {self._weights_path}")
                CNNProcessor._model_cnn_internal_instance = None # Asegurar que quede None si falla
            except Exception as e:
                print(f"!!!!!!!! CNN ERROR FATAL al cargar modelo/pesos Keras: {e} !!!!!!!!!")
                CNNProcessor._model_cnn_internal_instance = None # Asegurar que quede None si falla
        
        # Asignar el modelo (posiblemente None si falló la carga) a la instancia
        self.model_cnn = CNNProcessor._model_cnn_internal_instance

        if self.model_cnn is None:
            print("--- CNN ADVERTENCIA: El modelo Keras (self.model_cnn) no está cargado. Las predicciones fallarán. ---")

    @classmethod
    def get_instance(cls):
        """
        Método de clase para obtener la instancia Singleton de CNNProcessor.
        """
        if cls._instance is None:
            print("--- CNN DEBUG: Creando NUEVA instancia de CNNProcessor (Singleton) ---")
            cls._instance = cls()  # Esto llamará a __init__ una sola vez
        # else:
            # print("--- CNN DEBUG: Reutilizando instancia existente de CNNProcessor (Singleton) ---")
        return cls._instance

    def _preprocess_image(self, image_pil, target_size=(224, 224)):
        if not isinstance(image_pil, Image.Image):
            print("--- CNN ERROR: _preprocess_image esperaba un objeto PIL.Image ---")
            return None
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        image_pil = image_pil.resize(target_size)
        image_array = img_to_array(image_pil)
        image_array = np.expand_dims(image_array, axis=0)
        # Descomenta la siguiente línea si tu modelo fue entrenado con imágenes normalizadas a [0,1]
        # image_array = image_array / 255.0 
        return image_array
    
    def predict_from_image_file(self, image_file_object):
        if not self.model_cnn: # Verificar si el modelo se cargó correctamente
            print("--- CNN ERROR: Modelo no cargado. No se puede realizar la predicción.")
            return None, 0.0

        print(f"--- CNN DEBUG: Iniciando predicción para: {getattr(image_file_object, 'name', 'archivo_desconocido')} ---")
        try:
            # Asegurarse de que el puntero del archivo esté al inicio si ya fue leído
            if hasattr(image_file_object, 'seek') and callable(image_file_object.seek):
                image_file_object.seek(0)
            img_pil = Image.open(image_file_object)
        except Exception as e:
            print(f"--- CNN ERROR: No se pudo abrir la imagen con PIL: {e}")
            return None, 0.0

        processed_image_array = self._preprocess_image(img_pil)
        if processed_image_array is None:
            return None, 0.0
        
        try:
            predictions_array = self.model_cnn.predict(processed_image_array)
            # print(f"--- CNN DEBUG: Array de predicciones crudas: {predictions_array}")
        except Exception as e:
            print(f"--- CNN ERROR: Falló la predicción del modelo Keras (self.model_cnn.predict): {e}")
            return None, 0.0

        index_prediction = np.argmax(predictions_array[0])
        confidence = float(np.max(predictions_array[0]) * 100) # Convertir a float de Python

        predicted_desease_object = None
        try:
            # Asumiendo que cnn_prediction_index es único en tu modelo Desease
            predicted_desease_object = Desease.objects.get(cnn_prediction_index=index_prediction)
            print(f"--- CNN DEBUG: Predicción - Índice: {index_prediction}, Confianza: {confidence:.2f}%, Enfermedad: {predicted_desease_object.name_desease}")
        except Desease.DoesNotExist:
            print(f"--- CNN ADVERTENCIA: No se encontró Desease en BD para cnn_prediction_index: {index_prediction}")
        except Desease.MultipleObjectsReturned:
             print(f"--- CNN ADVERTENCIA: Múltiples Desease para cnn_prediction_index: {index_prediction}. Usando la primera.")
             predicted_desease_object = Desease.objects.filter(cnn_prediction_index=index_prediction).first()
        except Exception as e:
            print(f"--- CNN ERROR al buscar Desease por índice ({index_prediction}): {e}")

        return predicted_desease_object, confidence

    def convert_pil_to_django_image_file(self, pil_image, original_filename="processed_image.jpg"):
        # Este método podría no ser necesario aquí si predict_from_image_file toma el objeto archivo de Django
        # y devuelve el objeto Desease. La vista guardaría la imagen original subida.
        # Lo mantengo por si lo necesitas para algún otro propósito.
        if not isinstance(pil_image, Image.Image):
            print("--- CNN ERROR: convert_pil_to_django_image_file esperaba un objeto PIL.Image ---")
            return None
        img_io = BytesIO()
        # Determinar formato basado en el nombre del archivo original o default a JPEG
        file_format = 'JPEG'
        if original_filename.lower().endswith('.png'):
            file_format = 'PNG'
        
        pil_image.save(img_io, format=file_format)
        img_io.seek(0)
        
        content_type = f'image/{file_format.lower()}'
        
        image_file = InMemoryUploadedFile(
            img_io, 
            None, # field_name
            original_filename, 
            content_type,
            img_io.getbuffer().nbytes, 
            None # charset
        )
        return image_file