import sys
import os

# Add project root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Add subfolders explicitly if needed
sys.path.insert(0, os.path.join(ROOT, "pretrain"))
sys.path.insert(0, os.path.join(ROOT, "Evaluation"))
sys.path.insert(0, os.path.join(ROOT, "data"))