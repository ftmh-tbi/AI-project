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

class GeneticAlgorithm:

    def __init__(self, map_obj, agents, goals, goal_labels=None,
                 population_size=100, generations=200,
                 mutation_rate=0.1, crossover_rate=0.8):
        self.map = map_obj
        self.agents = agents
        self.goals = goals
        self.goal_labels = goal_labels or [f"G{i+1}" for i in range(len(goals))]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_agents = len(agents)
        self.num_goals = len(goals)
        self.path_cache = {}
    
    def calculate_path_cost(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[float]:

        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        temp_map = Map(self.map.grid)
        temp_map.start = start
        temp_map.goal = goal
        temp_map.z_cells = self.map.z_cells
        
        search_engine = SearchEngine(temp_map)
        result = search_engine.a_star_search()
        
        if result:
            _, cost, _ = result
            self.path_cache[cache_key] = cost
            return cost
        else:
            self.path_cache[cache_key] = None
            return None
    
    def evaluate_chromosome(self, chromosome: Chromosome):

        agent_costs: Dict[int, float] = {agent.id: 0.0 for agent in self.agents}
        agent_positions: Dict[int, Tuple[int, int]] = {
            agent.id: agent.start_position for agent in self.agents
        }
        
        for goal_idx, agent_id in enumerate(chromosome.genes):
            goal_pos = self.goals[goal_idx]
            current_pos = agent_positions[agent_id]
            
            cost = self.calculate_path_cost(current_pos, goal_pos)
            
            if cost is None:
                chromosome.makespan = float('inf')
                chromosome.fitness = 0.0
                return
            
            agent_costs[agent_id] += cost
            agent_positions[agent_id] = goal_pos
        
        chromosome.agent_costs = agent_costs
        chromosome.makespan = max(agent_costs.values())
        
        if chromosome.makespan > 0:
            chromosome.fitness = 1.0 / chromosome.makespan
        else:
            chromosome.fitness = float('inf')
    
    def create_random_chromosome(self) -> Chromosome:
        genes = [random.randint(0, self.num_agents - 1) for _ in range(self.num_goals)]
        return Chromosome(genes)
    
    def initialize_population(self) -> List[Chromosome]:
        population = []
        
        for _ in range(self.population_size):
            chromosome = self.create_random_chromosome()
            self.evaluate_chromosome(chromosome)
            population.append(chromosome)
        
        return population
    
    def selection(self, population: List[Chromosome]) -> Chromosome:
        tournament_size = 5
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda c: c.makespan)
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:

        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, self.num_goals - 1)
        
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        
        return Chromosome(child1_genes), Chromosome(child2_genes)
    
    def mutate(self, chromosome: Chromosome):

        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_rate:
                chromosome.genes[i] = random.randint(0, self.num_agents - 1)
    
    def run(self, verbose: bool = True) -> Tuple[Chromosome, List[float]]:

        population = self.initialize_population()
        best_makespan_history = []
        
        for generation in range(self.generations):
            population.sort(key=lambda c: c.makespan)
            best_chromosome = population[0]
            best_makespan_history.append(best_chromosome.makespan)
            
            if verbose and generation % 20 == 0:
                print(f"Generation {generation}: Best Makespan = {best_chromosome.makespan:.2f}")
            
            new_population = []
            
            elite_size = int(0.1 * self.population_size)
            new_population.extend([c.copy() for c in population[:elite_size]])
            
            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutate(child1)
                self.mutate(child2)
                
                self.evaluate_chromosome(child1)
                self.evaluate_chromosome(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        population.sort(key=lambda c: c.makespan)
        best_solution = population[0]
        
        if verbose:
            print(f"\nFinal Best Makespan: {best_solution.makespan:.2f}")
        
        return best_solution, best_makespan_history
    
    def print_solution(self, chromosome: Chromosome):
        print(f"Best makespan (minutes): {chromosome.makespan:.2f}")

        agent_goal_idxs = {agent.id: [] for agent in self.agents}
        for goal_idx, agent_id in enumerate(chromosome.genes):
            agent_goal_idxs[agent_id].append(goal_idx)

        for agent in self.agents:
            idxs = agent_goal_idxs[agent.id]
            labels = [self.goal_labels[g] for g in idxs]
            coords = [self.goals[g] for g in idxs]
            print(f"  S{agent.id + 1} at {agent.start_position} "
                  f"assigned targets: {labels} -> coords: {coords}")

        for agent in self.agents:
            cost = chromosome.agent_costs.get(agent.id, 0.0)
            print(f"  S{agent.id + 1} route time = {cost} minutes")

def parse_phase3_input(input_text: str):

    lines = input_text.strip().split('\n')
    n, m = map(int, lines[0].split())

    grid = []
    raw_agents = []  
    raw_goals = []    

    for i in range(1, n + 1):
        row = lines[i].split()
        grid.append(row)
        for j, cell in enumerate(row):
            if cell.startswith('S') and len(cell) > 1 and cell[1:].isdigit():
                agent_id = int(cell[1:]) - 1
                raw_agents.append((agent_id, (i - 1, j)))
            elif cell.startswith('G') and len(cell) > 1 and cell[1:].isdigit():
                gnum = int(cell[1:])
                raw_goals.append((gnum, (i - 1, j), cell))

    raw_agents.sort(key=lambda x: x[0])
    raw_goals.sort(key=lambda x: x[0])

    agents = [Agent(aid, pos) for aid, pos in raw_agents]
    goals = [pos for _, pos, _ in raw_goals]
    goal_labels = [label for _, _, label in raw_goals]

    return grid, agents, goals, goal_labels


def run_phase3_genetic(input_text: str, population_size: int = 100,
                       generations: int = 200, mutation_rate: float = 0.1):
    grid, agents, goals, goal_labels = parse_phase3_input(input_text)

    map_obj = Map(grid)

    ga = GeneticAlgorithm(
        map_obj,
        agents,
        goals,
        goal_labels,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate
    )

    best_solution, history = ga.run(verbose=False)

    ga.print_solution(best_solution)

    return best_solution, history
