# apps.py
from django.apps import AppConfig


class SotopiaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sotopia"

class ToMSamplerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = 'sotopia'
    tom_sampler = None  # Initialized in the custom command
