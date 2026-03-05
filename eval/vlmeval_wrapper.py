"""DocSP-InternVL wrapper for VLMEvalKit evaluation."""

import sys
import os

# Register DocSP custom model classes before any AutoModel usage
DOCSP_DIR = "/NetDisk/juyeon/DocSP"
sys.path.insert(0, DOCSP_DIR)

from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel
from transformers import AutoConfig, AutoModel

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

# Now import InternVLChat which uses AutoModel.from_pretrained
from .internvl import InternVLChat
