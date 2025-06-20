# config/settings.py (Para el proyecto chatbot simplificado)

from pathlib import Path
import environ
import os
# from datetime import timedelta # No necesitamos JWT si no hay login para el chatbot

# --- 1. Definición de BASE_DIR ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 2. Inicializar django-environ y LEER .env ---
env = environ.Env(
    DJANGO_DEBUG=(bool, True),
    DJANGO_ALLOWED_HOSTS=(list, ['127.0.0.1', 'localhost']),
    DB_ENGINE=(str, 'django.db.backends.sqlite3'), # Default a SQLite para simplicidad inicial
    DB_NAME_SQLITE=(str, str(BASE_DIR / 'db.sqlite3')), # Default nombre para SQLite
    OPENAI_API_KEY=(str, None),
    # No necesitamos FIELD_ENCRYPTION_KEY si no usamos la app Informativo o campos encriptados
)
ENV_FILE_PATH = os.path.join(BASE_DIR, '.env')
print(f"--- SETTINGS DEBUG: Intentando leer .env desde: {ENV_FILE_PATH} ---")
if os.path.exists(ENV_FILE_PATH):
    environ.Env.read_env(ENV_FILE_PATH)
    print(f"--- SETTINGS DEBUG: Archivo .env encontrado y leído. ---")
else:
    print(f"--- SETTINGS ADVERTENCIA: Archivo .env NO encontrado en {ENV_FILE_PATH}. ---")

# --- 3. Configuración ---
SECRET_KEY = env('DJANGO_SECRET_KEY', default="django-insecure-CAMBIAR_ESTA_CLAVE_EN_PRODUCCION_!@#secret")
DEBUG = env('DJANGO_DEBUG')
ALLOWED_HOSTS = env.list('DJANGO_ALLOWED_HOSTS_CSV', default=['127.0.0.1', 'localhost'])

# --- Aplicaciones Instaladas ---
INSTALLED_APPS = [
    'django.contrib.admin',      # Para el modelo KnownDesease
    'django.contrib.auth',       # Requerido por el admin
    'django.contrib.contenttypes',
    'django.contrib.sessions',   # Esencial para el chat anónimo
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'chatbot',                   # Nuestra app principal
    'widget_tweaks',           # Opcional para plantillas
   # 'rest_framework',          # No necesario si solo usamos formularios Django para este chatbot
]

# --- Middleware ---
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware', # Esencial
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware', # Bueno tenerlo para formularios POST
    'django.contrib.auth.middleware.AuthenticationMiddleware', # Para el admin
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls' # Asumiendo que tu proyecto se llama 'config'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'], # Para base.html
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth', # Para el admin
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application' # Asumiendo proyecto 'config'

# --- Base de Datos ---
DATABASES = {
    'default': {
        'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.postgresql'),
        'NAME': os.getenv('POSTGRES_DB', 'dermabot_01'),
        'USER': os.getenv('POSTGRES_USER', 'postgres'),
        'PASSWORD': os.getenv('POSTGRES_PASSWORD', 'tu_password'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# --- Autenticación (Primariamente para el Admin) ---
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
] # Simplificado un poco

# --- Internacionalización ---
LANGUAGE_CODE = 'es-ar'
TIME_ZONE = 'America/Argentina/Buenos_Aires'
USE_I18N = True
USE_TZ = True

# --- Archivos Estáticos y Media ---
STATIC_URL = 'static/'
# STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles_production') # Para producción
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # ¡CREA ESTA CARPETA 'media' EN LA RAÍZ!

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- API KEY de OpenAI ---
OPENAI_API_KEY = env('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("ADVERTENCIA: La variable OPENAI_API_KEY no está definida en .env")
