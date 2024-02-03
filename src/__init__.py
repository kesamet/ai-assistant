import logging

import box
import yaml
from dotenv import load_dotenv

with open("config.yaml", "r", encoding="utf8") as f:
    CFG = box.Box(yaml.safe_load(f))

_ = load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
