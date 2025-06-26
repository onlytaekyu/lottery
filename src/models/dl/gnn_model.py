"""
그래프 신경망 기반 모델 (Graph Neural Network Model)

이 모듈은 로또 번호 간의 연관성을 그래프로 모델링하여 예측하는 GNN 모델을 구현합니다.
번호 간 동시 출현 관계를 학습하여 향후 등장 가능성이 높은 번호 조합을 예측합니다.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import random

from ..base_model import ModelWithAMP
from ...utils.error_handler import get_logger

# 로거 설정
logger = get_logger(__name__)
