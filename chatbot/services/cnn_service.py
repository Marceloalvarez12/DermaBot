# chatbot/services/cnn_service.py
import os
import sys
from PIL import Image
from django.conf import settings
from keras.models import load_model # Asegúrate que esté importado
from io import BytesIO
from keras.preprocessing.image import img_to_array
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile
from ..models import Desease # Importa el modelo Desease unificado de la app chatbot


class CNNProcessor:
    _model_cnn_instance = None
    # Ajusta esta ruta si tu carpeta AI_Model está en otro lugar (ej. dentro de chatbot/AI_Model)
    _model_path = os.path.join(settings.BASE_DIR, 'chatbot', 'AI_Model', 'skin-cancer-7-classes_MobileNet_ph2_model.h5')
    _weights_path = os.path.join(settings.BASE_DIR, 'chatbot', 'AI_Model', 'skin-cancer-7-classes_MobileNet_ph2_weights.keras')

    def __init__(self):
        if CNNProcessor._model_cnn_instance is None:
            print("--- CNN DEBUG: Cargando modelo Keras y pesos por primera vez... ---")
            try:
                #################################################################
                # INICIO DEL CAMBIO IMPORTANTE (Solución 1)                     #
                #################################################################
                print(f"--- CNN DEBUG: Intentando cargar modelo desde: {self._model_path} con compile=False ---")
                CNNProcessor._model_cnn_instance = load_model(
                    self._model_path, 
                    custom_objects=None, 
                    compile=False  # <<--- CAMBIO A False AQUÍ
                )
                print(f"--- CNN DEBUG: Modelo cargado (compile=False). Intentando cargar pesos desde: {self._weights_path} ---")
                CNNProcessor._model_cnn_instance.load_weights(self._weights_path)
                print("--- CNN DEBUG: Modelo Keras y pesos cargados exitosamente (compile=False). ---")

                # Opcional: Si tu modelo necesita ser compilado para .predict() (aunque usualmente no si los pesos están cargados)
                # o si obtienes un error diferente después de este cambio, podrías necesitar recompilarlo.
                # Si es así, necesitarás el optimizador y la función de pérdida correctos.
                # Ejemplo genérico (ajusta según tu modelo):
                # print("--- CNN DEBUG: Recompilando modelo para predicción (opcional)... ---")
                # CNNProcessor._model_cnn_instance.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # print("--- CNN DEBUG: Modelo recompilado. ---")

                #################################################################
                # FIN DEL CAMBIO IMPORTANTE                                     #
                #################################################################
            except Exception as e:
                print(f"!!!!!!!! CNN ERROR FATAL al cargar modelo/pesos Keras: {e} !!!!!!!!!")
                raise # Re-lanza la excepción para que Django la maneje y veas el error completo
        self.model_cnn = CNNProcessor._model_cnn_instance

    def _preprocess_image(self, image_pil, target_size=(224, 224)):
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        image_pil = image_pil.resize(target_size)
        image_array = img_to_array(image_pil)
        image_array = np.expand_dims(image_array, axis=0)
        # image_array = image_array / 255.0 # Descomenta si tu modelo fue entrenado con esto
        return image_array
    
    def predict_from_image_file(self, image_file_object):
        print(f"--- CNN DEBUG: Iniciando predicción para: {getattr(image_file_object, 'name', 'archivo_desconocido')} ---")
        try:
            img_pil = Image.open(image_file_object)
        except Exception as e:
            print(f"--- CNN ERROR: No se pudo abrir la imagen con PIL: {e}")
            return None, 0.0

        processed_image_array = self._preprocess_image(img_pil)
        
        try:
            predictions_array = self.model_cnn.predict(processed_image_array)
        except Exception as e:
            print(f"--- CNN ERROR: Falló la predicción del modelo Keras: {e}")
            return None, 0.0

        index_prediction = np.argmax(predictions_array[0])
        confidence = float(np.max(predictions_array[0]) * 100)

        predicted_desease_object = None
        try:
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
        img_io = BytesIO()
        pil_image.save(img_io, format='JPEG')
        img_io.seek(0)
        image_file = InMemoryUploadedFile(
            img_io, None, original_filename, 'image/jpeg',
            img_io.getbuffer().nbytes, None
        )
        return image_file