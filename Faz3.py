import random
from typing import List, Tuple, Dict, Optional
from Faz1_2 import Map
from Faz1_2 import SearchEngine




class Agent:

    
    def __init__(self, agent_id: int, start_position: Tuple[int, int]):
        self.id = agent_id
        self.start_position = start_position
    
    def __repr__(self):
        return f"Agent({self.id}, {self.start_position})"
