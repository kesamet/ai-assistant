import box
import torch
import yaml
from dotenv import load_dotenv

with open("config.yaml", "r", encoding="utf8") as f:
    CFG = box.Box(yaml.safe_load(f))

_ = load_dotenv()

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    CFG.DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    CFG.DEVICE = torch.device("cuda")
else:
    CFG.DEVICE = torch.device("cpu")
