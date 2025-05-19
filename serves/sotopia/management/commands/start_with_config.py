# sotopia/management/commands/start_with_config.py
from django.core.management import execute_from_command_line
from django.core.management.base import BaseCommand
from sotopia.apps import RejectionSamplerConfig
from sotopia.models import RejectionSampler


class Command(BaseCommand):
    help = 'Start the server with custom RejectionSampler configuration'

    def add_arguments(self, parser):
        parser.add_argument('--base_model_path', required=True, type=str, help='Name of the SFT model')
        parser.add_argument('--grpo_model_path', required=True, type=str, help='VLLM API URL for the SFT model')
        parser.add_argument('--template_path', required=True, type=str, help='Path to the Reward model')
        parser.add_argument('--log_path', required=True, type=str, help='Name of the model')
        parser.add_argument('--port', type=int, default=8000, help='Port number for the Django server')

    def handle(self, *args, **options):
        # Set up the rejection sampler with the provided config
        config = {
            "base_model_path": options['base_model_path'],
            "grpo_model_path": options['grpo_model_path'],
            "template_path": options['template_path'],
            "log_path": options['log_path'],
        }

        # # Initialize the rejection_sampler directly
        RejectionSamplerConfig.rejection_sampler = RejectionSampler(**config)

        # Start the server with the specified port
        self.stdout.write(f"Starting the Django server on port {options['port']} with custom configuration...")
        execute_from_command_line(["hello", "runserver", "--noreload", f"{options['port']}"])
