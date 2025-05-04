# # sotopia/management/commands/start_with_config.py
# from django.core.management import execute_from_command_line
# from django.core.management.base import BaseCommand
# from sotopia.apps import RejectionSamplerConfig
# from sotopia.models import RejectionSampler


# class Command(BaseCommand):
#     help = 'Start the server with custom RejectionSampler configuration'

#     def add_arguments(self, parser):
#         parser.add_argument('--sft_model_name', required=True, type=str, help='Name of the SFT model')
#         parser.add_argument('--sft_model_vllm_api_url', required=True, type=str, help='VLLM API URL for the SFT model')
#         parser.add_argument('--reward_model_path', required=True, type=str, help='Path to the Reward model')
#         parser.add_argument('--reward_model_name', required=True, type=str, help='Name of the model')
#         parser.add_argument('--template_path', required=True, type=str, help='Path to the Jinja template')
#         parser.add_argument('--max_responses', type=int, default=5, help='Max responses')
#         parser.add_argument('--max_length', type=int, default=4096, help='Max length of responses')
#         parser.add_argument('--port', type=int, default=8000, help='Port number for the Django server')
#         parser.add_argument('--sft_batch_size', type=int, default=10, help='SFT batch size for the model')
#         parser.add_argument('--rm_batch_size', type=int, default=10, help='Reward model batch size for the model')

#     def handle(self, *args, **options):
#         # Set up the rejection sampler with the provided config
#         config = {
#             "sft_model_name": options['sft_model_name'],
#             "sft_model_vllm_api_url": options['sft_model_vllm_api_url'],
#             "reward_model_path": options['reward_model_path'],
#             "reward_model_name": options['reward_model_name'],
#             "template_path": options['template_path'],
#             "max_responses": options['max_responses'],
#             "max_length": options['max_length'],
#             "sft_batch_size": options['sft_batch_size'],
#             "rm_batch_size": options['rm_batch_size'],
#         }

#         # # Initialize the rejection_sampler directly
#         RejectionSamplerConfig.rejection_sampler = RejectionSampler(**config)

#         # Start the server with the specified port
#         self.stdout.write(f"Starting the Django server on port {options['port']} with custom configuration...")
#         execute_from_command_line(["hello", "runserver", "--noreload", f"{options['port']}"])


# sotopia/management/commands/start_with_tomsampler.py

import os
from django.core.management import execute_from_command_line
from django.core.management.base import BaseCommand
from sotopia.apps import SotopiaConfig
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
        tom_sampler = ToMSampler(
            model_path=options['model_path'],
            base_model=options['base_model'],
            template_path=options['template_path'],
            log_path=options['log_path'],
        )

        # Attach it to the global AppConfig so views and other code can access it:
        ToMSamplerConfig.tom_sampler = tom_sampler

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
        execute_from_command_line([
            'manage.py',
            'runserver',
            '--noreload',
            str(options['port']),
        ])
