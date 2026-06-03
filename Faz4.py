from typing import List, Tuple, Optional, Set
from Faz1_2 import Map
from Faz1_2 import parse_input
from Faz1_2 import PriorityQueue


def get_bridge_num(cell: str) -> Optional[int]:

    if cell.startswith('B'):
        try:
            return int(cell[1:])
        except ValueError:
            return None
    return None


def get_cell_cost_phase4(map_obj: Map, pos: Tuple[int, int],
                         current_bridge: Optional[int]) -> float:

    if not map_obj.is_valid_position(pos):
        return float('inf')

    x, y = pos
    cell = map_obj.grid[x][y]
    bridge_num = get_bridge_num(cell)

    if bridge_num is not None:
        if current_bridge == bridge_num:
            return 0.0
        return float(bridge_num)

    if cell == 'S' or cell.startswith('S') or cell == 'G' or cell.startswith('G'):
        return 1.0

    if cell.isdigit():
        return float(cell)

    return float('inf')

class BridgeNode:

    def __init__(self, position, g_cost=0.0, h_cost=0.0,
                 parent=None, action=None, current_bridge=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.action = action
        self.current_bridge = current_bridge

    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def get_state(self) -> Tuple:
        return (self.position[0], self.position[1], self.current_bridge)
def a_star_phase4(map_obj: Map):

    if not map_obj.start or not map_obj.goal:
        return None

    heuristic = lambda p: map_obj.manhattan_distance(p, map_obj.goal)

    sr, sc = map_obj.start
    start_bridge = get_bridge_num(map_obj.grid[sr][sc])

    start_node = BridgeNode(
        map_obj.start, g_cost=0.0,
        h_cost=heuristic(map_obj.start),
        current_bridge=start_bridge
    )

    frontier = PriorityQueue()
    frontier.push(start_node)
    visited: Set[Tuple] = set()
    expanded = 0

    while not frontier.is_empty():
        current = frontier.pop()

        if current.position == map_obj.goal:
            actions = []
            node = current
            while node.parent is not None:
                actions.append(node.action)
                node = node.parent
            actions.reverse()
            return actions, current.g_cost, expanded

        state = current.get_state()
        if state in visited:
            continue
        visited.add(state)
        expanded += 1

        for action, npos in map_obj.get_neighbors(current.position):
            move_cost = get_cell_cost_phase4(map_obj, npos, current.current_bridge)
            if move_cost == float('inf'):
                continue

            new_g = current.g_cost + move_cost
            new_bridge = get_bridge_num(map_obj.grid[npos[0]][npos[1]])

            new_node = BridgeNode(
                position=npos, g_cost=new_g,
                h_cost=heuristic(npos),
                parent=current, action=action,
                current_bridge=new_bridge
            )

            if new_node.get_state() not in visited:
                frontier.push(new_node)

    return None

class IDAStarSolver:

    def __init__(self, map_obj: Map):
        self.map = map_obj
        self.goal = map_obj.goal
        self.expanded = 0

    def heuristic(self, pos) -> float:
        return self.map.manhattan_distance(pos, self.goal)

    def search(self):
        """
        Returns:
            (actions, cost, expanded_nodes) یا None
        """
        if not self.map.start or not self.goal:
            return None

        start = self.map.start
        start_bridge = get_bridge_num(self.map.grid[start[0]][start[1]])

        cutoff = self.heuristic(start)
        self.expanded = 0

        while True:
            visited: Set[Tuple] = {(start, start_bridge)}
            path = [(start, None, 0.0, start_bridge)]

            result = self._dfs(path, visited, cutoff)

            if isinstance(result, list):
                actions = [step[1] for step in result if step[1] is not None]
                cost = result[-1][2]
                return actions, cost, self.expanded

            if result == float('inf'):
                return None 

            cutoff = result

    def _dfs(self, path: List, visited: Set, cutoff: float):

        pos, _, g, current_bridge = path[-1]
        f = g + self.heuristic(pos)

        if f > cutoff:
            return f

        if pos == self.goal:
            return path

        self.expanded += 1

        min_t = float('inf')

        for action, npos in self.map.get_neighbors(pos):
            move_cost = get_cell_cost_phase4(self.map, npos, current_bridge)
            if move_cost == float('inf'):
                continue

            ng = g + move_cost
            new_bridge = get_bridge_num(self.map.grid[npos[0]][npos[1]])
            state = (npos, new_bridge)

            if state in visited:
                continue

            visited.add(state)
            path.append((npos, action, ng, new_bridge))

            result = self._dfs(path, visited, cutoff)

            if isinstance(result, list):
                return result
            if result < min_t:
                min_t = result

            path.pop()
            visited.discard(state)

        return min_t
def run_phase4_astar(input_text: str):
    grid = parse_input(input_text)
    map_obj = Map(grid)
    result = a_star_phase4(map_obj)
    if result:
        actions, cost, expanded = result
        print(f"Cost: {int(cost)} min")
        print(f"Actions: {actions}")
        print(f"Expanded nodes: {expanded}")
    else:
        print("path not found! ")


def run_phase4_idastar(input_text: str):
    grid = parse_input(input_text)
    map_obj = Map(grid)
    solver = IDAStarSolver(map_obj)
    result = solver.search()
    if result:
        actions, cost, expanded = result
        print(f"Cost: {int(cost)} min")
        print(f"Actions: {actions}")
        print(f"Expanded nodes: {expanded}")
    else:
        print("path not found! ")


if __name__ == "__main__":

    first = input().strip()          
    rows = int(first.split()[0])     

    test = first + "\n"              
    for _ in range(rows):
        test += input() + "\n"       

    print("\n" + "=" * 50)
    run_phase4_idastar(test)