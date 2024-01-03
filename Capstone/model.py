from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPath

from .configuration_phi import PhiConfig
