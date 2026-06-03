from typing import List, Tuple, Optional, Set, Dict
import math


class Node:
    
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, 
                 h_cost: float = 0, parent: Optional['Node'] = None, 
                 action: Optional[str] = None, time_mod: int = 0):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.action = action
        self.time_mod = time_mod
    
    def f_cost(self) -> float:

        return self.g_cost + self.h_cost
    
    def get_state(self) -> Tuple:

        return (self.position[0], self.position[1], self.time_mod)
    
    def __lt__(self, other: 'Node') -> bool:
        
        if abs(self.f_cost() - other.f_cost()) > 1e-9:
            return self.f_cost() < other.f_cost()
        
        return self.g_cost < other.g_cost
    
    def __eq__(self, other: 'Node') -> bool:
        return self.get_state() == other.get_state()
    
    def __hash__(self):
        return hash(self.get_state())


class PriorityQueue:

    def __init__(self):
        self.elements: List[Node] = []
    
    def is_empty(self) -> bool:
        return len(self.elements) == 0
    
    def push(self, node: Node):
        self.elements.append(node)
        self.elements.sort(key=lambda n: n.f_cost())
    
    def pop(self) -> Node:
        if self.is_empty():
            raise IndexError("queue is empty")
        return self.elements.pop(0)
    
    def size(self) -> int:
        return len(self.elements)


class Map:
    
    def __init__(self, grid: List[List[str]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.start = None
        self.goal = None
        self.z_cells: Set[Tuple[int, int]] = set()
        self._parse_map()
    
    def _parse_map(self):
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S' or cell.startswith('S'):
                    self.start = (i, j)
                elif cell == 'G' or cell.startswith('G'):
                    self.goal = (i, j)
                elif cell == 'Z':
                    self.z_cells.add((i, j))
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols
    
    def get_cell_cost(self, pos: Tuple[int, int], time: float) -> float:
        
        if not self.is_valid_position(pos):
            return float('inf')
        
        x, y = pos
        cell = self.grid[x][y]
        
        if cell == 'S' or cell.startswith('S') or cell == 'G' or cell.startswith('G'):
            return 1.0
        
        if cell == 'Z':
            time_mod = int(time) % 30
            return 1.0 if time_mod < 15 else 15.0
        
        if cell.isdigit():
            return float(cell)
        
        if cell.startswith('B'):
            try:
                return float(cell[1:])
            except ValueError:
                return float('inf')
        
        return float('inf')
    

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        neighbors = []
        x, y = pos
    
        directions = {
        'RIGHT': (0, 1), 'LEFT': (0, -1),
        'UP': (-1, 0), 'DOWN': (1, 0)
        }
    
        if self.z_cells:
            directions['STAY'] = (0, 0)
        
        for action, (dx, dy) in directions.items():
            new_pos = (x + dx, y + dy)
            if self.is_valid_position(new_pos):
                neighbors.append((action, new_pos))
        
        return neighbors

    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class SearchEngine:
    
    def __init__(self, map_obj: Map):
        self.map = map_obj
        self.expanded_nodes = 0
    
    def reconstruct_path(self, node: Node) -> Tuple[List[str], float]:
        actions = []
        current = node
        
        while current.parent is not None:
            if current.action:
                actions.append(current.action)
            current = current.parent
        
        actions.reverse()
        return actions, node.g_cost
    
    def uniform_cost_search(self) -> Optional[Tuple[List[str], float, int]]:
        if not self.map.start or not self.map.goal:
            return None
        
        start_node = Node(self.map.start, g_cost=0, time_mod=0)
        frontier = PriorityQueue()
        frontier.push(start_node)
        
        visited: Set[Tuple] = set()
        self.expanded_nodes = 0
        
        while not frontier.is_empty():
            current = frontier.pop()
            
            if current.position == self.map.goal:
                actions, cost = self.reconstruct_path(current)
                return actions, cost, self.expanded_nodes
            
            state = current.get_state()
            if state in visited:
                continue
            
            visited.add(state)
            self.expanded_nodes += 1
            
            for action, new_pos in self.map.get_neighbors(current.position):
                current_time = current.g_cost
                
                move_cost = self.map.get_cell_cost(new_pos, current_time)

                if move_cost == float('inf'):
                    continue
                
                new_g_cost = current.g_cost + move_cost
                new_time_mod = int(new_g_cost) % 30
                
                new_node = Node(
                    position=new_pos,
                    g_cost=new_g_cost,
                    parent=current,
                    action=action,
                    time_mod=new_time_mod
                )
                
                if new_node.get_state() not in visited:
                    frontier.push(new_node)
        
        return None
    
    def a_star_search(self, heuristic_func=None) -> Optional[Tuple[List[str], float, int]]:
        if not self.map.start or not self.map.goal:
            return None
        
        if heuristic_func is None:
            heuristic_func = lambda pos: self.map.manhattan_distance(pos, self.map.goal)
        
        start_node = Node(
            self.map.start, 
            g_cost=0, 
            h_cost=heuristic_func(self.map.start),
            time_mod=0
        )
        frontier = PriorityQueue()
        frontier.push(start_node)
        
        visited: Set[Tuple] = set()
        self.expanded_nodes = 0
        
        while not frontier.is_empty():
            current = frontier.pop()
            
            if current.position == self.map.goal:
                actions, cost = self.reconstruct_path(current)
                return actions, cost, self.expanded_nodes
            
            state = current.get_state()
            if state in visited:
                continue
            
            visited.add(state)
            self.expanded_nodes += 1
            
            for action, new_pos in self.map.get_neighbors(current.position):
                current_time = current.g_cost
                
                move_cost = self.map.get_cell_cost(new_pos, current_time)

                if move_cost == float('inf'):
                    continue
                
                new_g_cost = current.g_cost + move_cost
                new_h_cost = heuristic_func(new_pos)
                new_time_mod = int(new_g_cost) % 30
                
                new_node = Node(
                    position=new_pos,
                    g_cost=new_g_cost,
                    h_cost=new_h_cost,
                    parent=current,
                    action=action,
                    time_mod=new_time_mod
                )
                
                if new_node.get_state() not in visited:
                    frontier.push(new_node)
        
        return None


def parse_input(input_text: str) -> List[List[str]]:
    
    lines = input_text.strip().split('\n')
    n, m = map(int, lines[0].split())
    
    grid = []
    for i in range(1, n + 1):
        row = lines[i].split()
        grid.append(row)
    
    return grid


def run_phase1_ucs(input_text: str):
    
    print("=== phase1: Uniform Cost Search ===\n")
    
    grid = parse_input(input_text)
    map_obj = Map(grid)
    search_engine = SearchEngine(map_obj)
    
    result = search_engine.uniform_cost_search()
    
    if result:
        actions, cost, expanded = result
        print(f"Cost: {int(cost)} min")
        print(f"Actions: {actions}")
        print(f"Expanded nodes: {expanded}")
    else:
        print("path not found!")


def run_phase2_astar(input_text: str):
    
    print("\n=== phase2: A* Search ===\n")
    
    grid = parse_input(input_text)
    map_obj = Map(grid)
    search_engine = SearchEngine(map_obj)
    
    result = search_engine.a_star_search()
    
    if result:
        actions, cost, expanded = result
        print(f"Cost: {int(cost)} min")
        print(f"Actions: {actions}")
        print(f"Expanded nodes: {expanded}")
    else:
        print("path not found!")



if __name__ == "__main__":
    print("enter input:\n")

    first = input().strip()         
    rows = int(first.split()[0])  

    test = first + "\n"            
    for _ in range(rows):
        test += input() + "\n"      

    print("\n" + "=" * 50)
    run_phase1_ucs(test)
    run_phase2_astar(test)