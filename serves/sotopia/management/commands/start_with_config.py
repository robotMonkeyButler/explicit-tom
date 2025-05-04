# sotopia/management/commands/start_with_tomsampler.py
import os
from django.core.management import execute_from_command_line
from django.core.management.base import BaseCommand
from sotopia.apps import ToMSamplerConfig
from sotopia.models import ToMSampler

class Command(BaseCommand):
    help = 'Launch Django with a custom ToMSampler for GRPO model outputs'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model_path',
            required=True,
            help='Path to the local GRPO model directory (base + optional LoRA adapter)'
        )
        parser.add_argument(
            '--base_model',
            required=True,
            help='HuggingFace tokenizer identifier or local path (often same as model_path)'
        )
        parser.add_argument(
            '--template_path',
            required=True,
            help='Filesystem path to the Jinja2 template for prompt formatting'
        )
        parser.add_argument(
            '--log_path',
            default='raw_outputs.jsonl',
            help='Path to the JSONL file where raw model outputs will be appended'
        )
        parser.add_argument(
            '--port',
            type=int,
            default=8000,
            help='Port for Django development server'
        )

    def handle(self, *args, **options):
        # Instantiate the ToMSampler with the provided parameters

        config = {
            "model_path": options['model_path'],
            "base_model": options['base_model'],
            "template_path": options['template_path'],
            "log_path": options['log_path'],
        }

        # Attach it to the global AppConfig so views and other code can access it:
        ToMSamplerConfig.tom_sampler = ToMSampler(**config)

        self.stdout.write(
            self.style.SUCCESS(
                f"ToMSampler initialized:\n"
                f"  Model directory: {options['model_path']}\n"
                f"  Base model: {options['base_model']}\n"
                f"  Template: {options['template_path']}\n"
                f"  Raw-output log: {options['log_path']}\n"
                f"Starting Django server on port {options['port']}..."
            )
        )

        # Finally, launch the Django development server
        execute_from_command_line(["hello", "runserver", "--noreload", f"{options['port']}"])
