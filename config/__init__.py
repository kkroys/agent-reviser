import os
import yaml
from dotenv import load_dotenv

load_dotenv()

with open('config/config.yaml', 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

config = {
    **yaml_config,
    'env': {
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
}
