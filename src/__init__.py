from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

_ = load_dotenv()
try:
    CFG = OmegaConf.load("config.yaml")
except FileNotFoundError as e:
    logger.warning(e)
