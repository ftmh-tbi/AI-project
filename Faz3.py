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
class Chromosome:
    
    def __init__(self, genes: List[int]):

        self.genes = genes
        self.fitness = 0.0
        self.makespan = float('inf')
        self.agent_costs: Dict[int, float] = {}  
    
    def __repr__(self):
        return f"Chromosome(genes={self.genes}, makespan={self.makespan:.2f})"
    
    def copy(self) -> 'Chromosome':
        new_chrom = Chromosome(self.genes.copy())
        new_chrom.fitness = self.fitness
        new_chrom.makespan = self.makespan
        new_chrom.agent_costs = self.agent_costs.copy()
        return new_chrom
