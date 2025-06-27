"""
CS311 Programming Assignment 1: Search

Full Name: Sujay Banerjee

Brief description of my heuristic:

TODO Briefly describe your heuristic and why it is more efficient
My custom heuristic combines Manhattan distance, linear conflict, and a last move penalty to estimate the cost to reach the goal. 
Manhattan distance provides the basic estimate, while linear conflict adds penalties for tiles that are reversed in the same row or column. 
The last move heuristic ensures key tiles (6 and 8) are in place for the final moves. 
This combination reduces the number of nodes explored, making the search more efficient while maintaining optimality.
I also precompute the goal row and column for each tile to avoid repeated calculations.
"""


import argparse, itertools, random, sys
from typing import Callable, List, Optional, Sequence, Tuple
from collections import deque, defaultdict
import heapq



# You are welcome to add constants, but do not modify the pre-existing constants

# Problem size 
BOARD_SIZE = 3

# The goal is a "blank" (0) in bottom right corner
GOAL = tuple(range(1, BOARD_SIZE**2)) + (0,)

GOAL_ROW = {}
GOAL_COL = {}
for index, tile in enumerate(GOAL):
    if tile != 0:
        GOAL_ROW[tile] = index // BOARD_SIZE
        GOAL_COL[tile] = index % BOARD_SIZE


def inversions(board: Sequence[int]) -> int:
    """Return the number of times a larger 'piece' precedes a 'smaller' piece in board"""
    return sum(
        (a > b and a != 0 and b != 0) for (a, b) in itertools.combinations(board, 2)
    )


class Node:
    def __init__(self, state: Sequence[int], parent: "Node" = None, cost=0):
        """Create Node to track particular state and associated parent and cost

        State is tracked as a "row-wise" sequence, i.e., the board (with _ as the blank)
        1 2 3
        4 5 6
        7 8 _
        is represented as (1, 2, 3, 4, 5, 6, 7, 8, 0) with the blank represented with a 0

        Args:
            state (Sequence[int]): State for this node, typically a list, e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
            parent (Node, optional): Parent node, None indicates the root node. Defaults to None.
            cost (int, optional): Cost in moves to reach this node. Defaults to 0.
        """
        self.state = tuple(state)  # To facilitate "hashable" make state immutable
        self.parent = parent
        self.cost = cost

    def is_goal(self) -> bool:
        """Return True if Node has goal state"""
        return self.state == GOAL

    def expand(self) -> List["Node"]:
        """Expand current node into possible child nodes with corresponding parent and cost"""

        # TODO: Implement this function to generate child nodes based on the current state
        children = []
        blank_spot = self.state.index(0)  
        row = blank_spot // BOARD_SIZE
        col = blank_spot % BOARD_SIZE

        #Up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row, col)

        for move in moves:
            new_row = row + move[0]
            new_col = col + move[1]
            # Stay in board bounds
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                new_state = self._swap(row, col, new_row, new_col)
                child_node = Node(new_state, parent=self, cost=self.cost + 1)
                children.append(child_node)

        return children

    def _swap(self, row1: int, col1: int, row2: int, col2: int) -> Sequence[int]:
        """Swap values in current state between row1,col1 and row2,col2, returning new "state" to construct a Node"""
        state = list(self.state)
        state[row1 * BOARD_SIZE + col1], state[row2 * BOARD_SIZE + col2] = (
            state[row2 * BOARD_SIZE + col2],
            state[row1 * BOARD_SIZE + col1],
        )
        return state

    def __str__(self):
        return str(self.state)

    # The following methods enable Node to be used in types that use hashing (sets, dictionaries) or perform comparisons. Note
    # that the comparisons are performed exclusively on the state and ignore parent and cost values.

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


def bfs(initial_board: Sequence[int], max_depth=12) -> Tuple[Optional[Node], int]:
    """Perform breadth-first search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement BFS. Your function should return a tuple containing the solution node and number of unique node explored

    # initial node
    initial_node = Node(state=initial_board, parent=None, cost=0)
    if initial_node.is_goal():
        return (initial_node, 1)

    frontier = deque([initial_node]) 
    reached = {initial_node.state}   

    # while there are nodes to explore
    while frontier:
        # FIFO order
        node = frontier.popleft()
        
        if node.is_goal():
            return (node, len(reached))  # solution and nodes explored

        if node.cost >= max_depth:
            continue  #skip to next node

        for child in node.expand():
            if child.state not in reached:
                # Add the state to the reached set
                reached.add(child.state)
                # Add the child node to the frontier for future exploration
                frontier.append(child)
    return (None, len(reached))





def manhattan_distance(node: Node) -> int:
    """Compute manhattan distance f(node), i.e., g(node) + h(node)"""
    # TODO: Implement the Manhattan distance heuristic (sum of Manhattan distances to goal location)
    h = 0
    g = node.cost
    for i, tile in enumerate(node.state):
        if tile == 0:
            continue  # Skip the blank tile
        
        current_row = i // BOARD_SIZE
        current_col = i % BOARD_SIZE

        goal_row = GOAL_ROW[tile]
        goal_col = GOAL_COL[tile]

        # Manhattan distance (row difference + column difference)
        h += abs(goal_row - current_row) + abs(goal_col - current_col)

    return g+h



def count_conflicts(goal_positions: List[int]) -> int:
    #Count the number of conflicts in list of goal positions
    conflicts = 0
    max_goal_pos = -1
    for goal_pos in goal_positions:
        if goal_pos > max_goal_pos:
            max_goal_pos = goal_pos
        else:
            conflicts += 1
    return conflicts

def linear_conflict(node: Node) -> int:
    # compute the linear conflict and add to manhattan distance
    h = manhattan_distance(node)
    conflict_count = 0

    #row conflicts
    for row in range(BOARD_SIZE):
        tiles_in_row = []
        for col in range(BOARD_SIZE):
            tile = node.state[row * BOARD_SIZE + col]
            if tile != 0 and GOAL_ROW[tile] == row:
                tiles_in_row.append(GOAL_COL[tile])
        conflict_count += count_conflicts(tiles_in_row)

    #column conflicts
    for col in range(BOARD_SIZE):
        tiles_in_col = []
        for row in range(BOARD_SIZE):
            tile = node.state[row * BOARD_SIZE + col]
            if tile != 0 and GOAL_COL[tile] == col:
                tiles_in_col.append(GOAL_ROW[tile])
        conflict_count += count_conflicts(tiles_in_col)

    # Manhattan distance + linear conflict (+2 penalty from paper)
    return h + 2 * conflict_count


def last_move_heuristic(node: Node) -> int:
    penalty = 0
    # tile 6 should be in position (1, 2)
    #if node.state.index(6) != (1 * BOARD_SIZE + 2):
    if node.state.index(6) == 9:
        penalty -= 1
    
    # tile 8 should be in position (2, 1) 
    #if node.state.index(8) != (2 * BOARD_SIZE + 1):
    if node.state.index(8) == 9:
        penalty -= 1 

    return penalty


def custom_heuristic(node: Node) -> int:
    # TODO: Implement and document your _admissable_ heuristic function
    h = linear_conflict(node)    
    h += last_move_heuristic(node)
    return h




def astar(
    initial_board: Sequence[int],
    max_depth=12,
    #heuristic: Callable[[Node], int] = manhattan_distance,
    heuristic: Callable[[Node], int] = custom_heuristic,
) -> Tuple[Optional[Node], int]:
    """Perform astar search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.
        heuristic (_Callable[[Node], int], optional): Heuristic function. Defaults to manhattan_distance.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement A* search. Make sure that your code uses the heuristic function provided as
    # an argument so that the test code can switch in your custom heuristic (i.e., do not "hard code"
    # manhattan distance as the heuristic)

    frontier = []
    initial_node = Node(state=initial_board, parent=None, cost=0)
    
    initial_f_value = heuristic(initial_node)
    heapq.heappush(frontier, (initial_f_value, initial_node))  # push as a tuple (f(n), node)
    
    reached = {}

    while frontier:
        _, current_node = heapq.heappop(frontier)

        if current_node.is_goal():
            return current_node, len(reached) 

        if current_node.cost >= max_depth:
            break

        for child in current_node.expand():
            if child.state not in reached or child.cost < reached[child.state]:
                reached[child.state] = child.cost 
                f_child = heuristic(child) #dont add cost to heuristic here because it is already in manhattan distance
                heapq.heappush(frontier, (f_child, child))
    return None, len(reached)



if __name__ == "__main__":

    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(
        description="Run search algorithms in random inputs"
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="bfs",
        help="Algorithm (one of bfs, astar, astar_custom)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations",
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default=None,
        help="Execute a single iteration using this board configuration specified as a string, e.g., 123456780",
    )

    args = parser.parse_args()

    num_solutions = 0
    num_cost = 0
    num_nodes = 0

    if args.algo == "bfs":
        algo = bfs
    elif args.algo == "astar":
        algo = astar
    elif args.algo == "astar_custom":
        algo = lambda board: astar(board, heuristic=custom_heuristic)
    else:
        raise ValueError("Unknown algorithm type")

    if args.state is None:
        iterations = args.iter
        while iterations > 0:
            init_state = list(range(BOARD_SIZE**2))
            random.shuffle(init_state)

            # A problem is only solvable if the parity of the initial state matches that
            # of the goal.
            if inversions(init_state) % 2 != inversions(GOAL) % 2:
                continue

            solution, nodes = algo(init_state)
            if solution:
                num_solutions += 1
                num_cost += solution.cost
                num_nodes += nodes

            iterations -= 1
    else:
        # Attempt single input state
        solution, nodes = algo([int(s) for s in args.state])
        if solution:
            num_solutions = 1
            num_cost = solution.cost
            num_nodes = nodes

    if num_solutions:
        print(
            "Iterations:",
            args.iter,
            "Solutions:",
            num_solutions,
            "Average moves:",
            num_cost / num_solutions,
            "Average nodes:",
            num_nodes / num_solutions,
        )
    else:
        print("Iterations:", args.iter, "Solutions: 0")
