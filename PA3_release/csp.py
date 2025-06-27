"""
CS311 Programming Assignment 3: CSP

Full Name: Sujay Banerjee

Brief description of my solver:
In my custom backtracking search algorithm, I implemented forward checking to reduce the domain of the neighbors of the assigned variable. 
The algorithm uses the most constrained variable heuristic to select the next variable to assign a value to. 
It then uses the least constraining value heuristic to select the value to assign to the variable. 
The algorithm also uses AC3 to enforce arc consistency before starting the search. 
TODO Briefly describe your solver. Did it perform better than the AC3 solver? If so, why do you think so? If not, can you think of any ways to improve it?
"""

import argparse, time
from functools import wraps
from typing import Dict, List, Optional, Set, Tuple

# You are welcome to add constants, but do not modify the pre-existing constants

# Length of side of a Soduku board
SIDE = 9

# Length of side of "box" within a Soduku board
BOX = 3

# Domain for cells in Soduku board
DOMAIN = range(1, 10)

# Helper constant for checking a Soduku solution
SOLUTION = set(DOMAIN)


def check_solution(board: List[int], original_board: List[int]) -> bool:
    """Return True if board is a valid Sudoku solution to original_board puzzle"""
    # print(f"check_solution called with board: {board}, original_board: {original_board}")
    # Original board values are maintained
    for s, o in zip(board, original_board):
        if o != 0 and s != o:
            # print(f"Mismatch at index {idx}: original {o}, solution {s}")
            return False
    for i in range(SIDE):
        # Valid row
        row = set(board[i * SIDE : (i + 1) * SIDE])
        if set(board[i * SIDE : (i + 1) * SIDE]) != SOLUTION:
            # print(f"Invalid row at index {i}: {row}")
            return False
        # Valid column
        column = set(board[i : SIDE * SIDE : SIDE])
        if set(board[i : SIDE * SIDE : SIDE]) != SOLUTION:
            # print(f"Invalid column at index {i}: {column}")
            return False
        # Valid Box (here i is serving as the "box" id since there are SIDE boxes)
        box_row, box_col = (i // BOX) * BOX, (i % BOX) * BOX
        box = set()
        for r in range(box_row, box_row + BOX):
            box.update(board[r * SIDE + box_col : r * SIDE + box_col + BOX])
        if box != SOLUTION:
            # print(f"Invalid box at index {i}: {box}")
            return False
    return True


def countcalls(func):
    """Decorator to track the number of times a function is called. Provides `calls` attribute."""
    countcalls.calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        initial_calls = countcalls.calls
        countcalls.calls += 1
        result = func(*args, **kwargs)
        wrapper.calls = countcalls.calls - initial_calls
        return result

    return wrapper



# The @countcalls decorator tracks the number of times we call the recursive function. Make sure the decorator
# is included on your recursive search function if you change the implementation.
@countcalls
def backtracking_search(
    neighbors: List[List[int]],
    queue: Set[Tuple[int, int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """Perform backtracking search on CSP using AC3

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable
        assignment (Dict[int, int]): Current variable->value assignment 

    Returns:
        Optional[Dict[int, int]]: Solution or None indicating no solution found
    """
    # TODO: Implement the backtracking search algorithm here


    # Check if all cells are assigned a value
    if len(assignment) == SIDE * SIDE:
        # print(f"Solution found with Assignment: {assignment}")
        return assignment

    # Select an unassigned variable with the smallest domain
    unassigned_vars = [var for var in range(SIDE * SIDE) if var not in assignment]
    var = min(unassigned_vars, key=lambda var: len(domains[var]))

    # Iterate through possible values in the domain of the selected variable
    for value in domains[var]:
        # Check if value is consistent with the assignment
        if is_consistent(var, value, assignment, neighbors):
            assignment[var] = value

            undo_log = []

            # Reduce the domain of var to the assigned value
            original_domain = domains[var][:] 
            domains[var] = [value]
            undo_log.append(('assign', var, original_domain)) # Record the assignment for backtracking

            # Initialize a new local queue for AC3
            local_queue = set()
            for neighbor in neighbors[var]:
                local_queue.add((neighbor, var))

            # Call AC3 to enforce arc consistency
            if ac3(local_queue, neighbors, domains, undo_log):
                # Proceed with recursive call if AC3 succeeds
                result = backtracking_search(neighbors, queue, domains, assignment)
                if result is not None:
                    return result
       
            # If AC3 fails or recursion doesn't find a solution, backtrack
            # Restore domains using undo log
            for action, var_to_restore, value_to_restore in reversed(undo_log):
                if action == 'assign':
                    domains[var_to_restore] = value_to_restore
                elif action == 'remove':
                    domains[var_to_restore].append(value_to_restore)
            del assignment[var]

    return None


def is_consistent(var: int, value: int, assignment: Dict[int, int], neighbors: List[List[int]]) -> bool:
    # Check if the value is consistent with the assignment
    for neighbor in neighbors[var]:
        if neighbor in assignment and assignment[neighbor] == value:
            return False
    return True


def ac3(queue: Set[Tuple[int, int]], neighbors: List[List[int]], domains: List[List[int]], undo_log: List[Tuple[int, int]]) -> bool:
    #return True if arcs are consistent
    while queue:
        (xi, xj) = queue.pop()
        if revise(domains, xi, xj, undo_log):
            if not domains[xi]:
                return False  # Domain empty
            for xk in neighbors[xi]:
                if xk != xj:
                    queue.add((xk, xi))
    return True

def revise(domains: List[List[int]], xi: int, xj: int, undo_log: List[Tuple[int, int]]) -> bool:
    #return True if we revise the domain
    revised = False
    if len(domains[xj]) == 1:
        value = domains[xj][0]
        if value in domains[xi]: 
            domains[xi].remove(value)
            undo_log.append(('remove', xi, value))  # Record the removal for backtracking
            revised = True
    return revised


def sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using backtracking search with the AC3 algorithm

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and
            a count of calls to recursive backtracking function
    """

    # print(f"Starting Sudoku Solver with board: {board}")

    domains = [[val] if val else list(DOMAIN) for val in board]
    neighbors = []
    queue = set()

    # TODO: Complete the initialization of the neighbors and queue data structures
    for i in range(SIDE * SIDE):
        row = i // SIDE
        col = i % SIDE
        box_row = (row // BOX) * BOX
        box_col = (col // BOX) * BOX

        # cells in the same row
        row_neighbors = [row * SIDE + c for c in range(SIDE) if c != col]

        # cells in the same column
        col_neighbors = [r * SIDE + col for r in range(SIDE) if r != row]

        # Indices of cells in the same box
        box_neighbors = []
        for r in range(box_row, box_row + BOX):
            for c in range(box_col, box_col + BOX):
                idx = r * SIDE + c
                if idx != i:
                    box_neighbors.append(idx)

        # Add neighbors to the list
        unique_neighbors = set(row_neighbors + col_neighbors + box_neighbors)
        neighbors.append(list(unique_neighbors))

    # add arcs to the queue
    for xi in range(SIDE * SIDE):
        for xj in neighbors[xi]:
            queue.add((xi, xj))

    # print(f"Initial domains: {domains}")
    
    # Initialize the assignment for any squares with domains of size 1 (e.g., pre-specified squares).
    # While not necessary for correctness, initializing the assignment improves performance, especially
    # for plain backtracking search.
    assignment = {
        var: domain[0] for var, domain in enumerate(domains) if len(domain) == 1
    }

    result = backtracking_search(neighbors, queue, domains, assignment)

    # Convert result dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, backtracking_search.calls





@countcalls
def my_backtracking_search(
    neighbors: List[List[int]],
    queue: Set[Tuple[int, int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """Custom backtracking search implementing efficient heuristics

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable
        assignment (Dict[int, int]): Current variable->value assignment 

    Returns:
        Optional[Dict[int, int]]: Solution or None indicating no solution found
    """

    # Check if all cells are assigned a value
    if len(assignment) == SIDE * SIDE:
        # print(f"Solution found with Assignment: {assignment}")
        return assignment

#    most constrained variable heuristic
    unassigned_vars = [var for var in range(SIDE * SIDE) if var not in assignment]
    var = min(unassigned_vars, key=lambda var: len(domains[var]))

#     # least constraining value heuristic
    values = sorted(domains[var], key=lambda val: sum(val in domains[neighbor] for neighbor in neighbors[var]))

    # Iterate through possible values in the domain of the selected variable
    for value in values:
        # Check if value is consistent with the assignment
        if is_consistent(var, value, assignment, neighbors):
            # Assign the value to the variable
            assignment[var] = value
            
            # Initialize the undo log
            undo_log = []

            # Reduce the domain of var to the assigned value
            undo_log.extend((var, val) for val in domains[var] if val != value)
            domains[var] = [value]

            

            if forward_check(var, value, neighbors, domains, undo_log):   
                result = my_backtracking_search(neighbors, queue, domains, assignment)
                if result is not None:
                    return result
       
            # Restore domains using undo log
            for var_to_restore, values_to_restore in undo_log:
                domains[var_to_restore].append(values_to_restore)
            del assignment[var]

    return None


def forward_check(var: int, value: int, neighbors: List[List[int]], domains: List[List[int]], undo_log: List[Tuple[int, int]]) -> bool:
    # For each neighbor of var, remove value from its domain
    for neighbor in neighbors[var]:
        if value in domains[neighbor]:
            domains[neighbor].remove(value)
            undo_log.append((neighbor, value))
            if len(domains[neighbor]) == 0:  # Domain empty
                return False
    return True





def my_sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using your own custom solver

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and
            a count of calls to recursive backtracking function
    """

    domains = [[val] if val else list(DOMAIN) for val in board]
    neighbors = []
    queue = set()

    # TODO: Complete the initialization of the neighbors and queue data structures

    for i in range(SIDE * SIDE):
        row = i // SIDE
        col = i % SIDE
        box_row = (row // BOX) * BOX
        box_col = (col // BOX) * BOX

        # cells in the same row
        row_neighbors = [row * SIDE + c for c in range(SIDE) if c != col]

        # cells in the same column
        col_neighbors = [r * SIDE + col for r in range(SIDE) if r != row]

        # Indices of cells in the same box
        box_neighbors = []
        for r in range(box_row, box_row + BOX):
            for c in range(box_col, box_col + BOX):
                idx = r * SIDE + c
                if idx != i:
                    box_neighbors.append(idx)

        # Add neighbors to the list
        unique_neighbors = set(row_neighbors + col_neighbors + box_neighbors)
        neighbors.append(list(unique_neighbors))

    # add arcs to the queue
    for xi in range(SIDE * SIDE):
        for xj in neighbors[xi]:
            queue.add((xi, xj))

    
    # Initialize the assignment for any squares with domains of size 1 (e.g., pre-specified squares).
    assignment = {
        var: domain[0] for var, domain in enumerate(domains) if len(domain) == 1
    }

    # Run AC3 before starting the search
    ac3(queue, neighbors, domains, [])
    

    result = my_backtracking_search(neighbors, queue, domains, assignment)

    # Convert assignment dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, my_backtracking_search.calls


if __name__ == "__main__":
    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(description="Run sudoku solver")
    parser.add_argument(
        "-a",
        "--algo",
        default="ac3",
        help="Algorithm (one of ac3, custom)",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="easy",
        help="Difficulty level (one of easy, medium, hard)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        type=int,
        help="Number of trials for timing",
    )
    parser.add_argument("puzzle", nargs="?", type=str, default=None)

    args = parser.parse_args()

    # fmt: off
    if args.puzzle:
        board = [int(c) for c in args.puzzle]
        if len(board) != SIDE*SIDE or set(board) > (set(DOMAIN) | { 0 }):
            raise ValueError("Invalid puzzle specification, it must be board length string with digits 0-9")
    elif args.level == "easy":
        board = [
            0,0,0,1,3,0,0,0,0,
            7,0,0,0,4,2,0,8,3,
            8,0,0,0,0,0,0,4,0,
            0,6,0,0,8,4,0,3,9,
            0,0,0,0,0,0,0,0,0,
            9,8,0,3,6,0,0,5,0,
            0,1,0,0,0,0,0,0,4,
            3,4,0,5,2,0,0,0,8,
            0,0,0,0,7,3,0,0,0,
        ]
    elif args.level == "medium":
        board = [
            0,4,0,0,9,8,0,0,5,
            0,0,0,4,0,0,6,0,8,
            0,5,0,0,0,0,0,0,0,
            7,0,1,0,0,9,0,2,0,
            0,0,0,0,8,0,0,0,0,
            0,9,0,6,0,0,3,0,1,
            0,0,0,0,0,0,0,7,0,
            6,0,2,0,0,7,0,0,0,
            3,0,0,8,4,0,0,6,0,
        ]
    elif args.level == "hard":
        board = [
            1,2,0,4,0,0,3,0,0,
            3,0,0,0,1,0,0,5,0,  
            0,0,6,0,0,0,1,0,0,  
            7,0,0,0,9,0,0,0,0,    
            0,4,0,6,0,3,0,0,0,    
            0,0,3,0,0,2,0,0,0,    
            5,0,0,0,8,0,7,0,0,    
            0,0,7,0,0,0,0,0,5,    
            0,0,0,0,0,0,0,9,8,
        ]
    elif args.level == "unsolvable":
        board = [
            2,5,6,0,4,9,8,3,7,
            1,8,3,5,0,7,9,6,4,
            9,7,4,3,8,6,2,5,1,
            8,4,9,1,6,2,3,7,5,
            5,6,2,7,9,3,4,1,8,
            7,3,1,4,5,8,6,2,9,
            6,9,7,8,3,1,5,4,2,
            4,2,8,6,7,5,1,9,3,
            3,1,5,9,2,4,7,8,6,
        ]
    elif args.level == "unsolvable_hard":
        board = [
            0,0,1,0,0,0,0,0,0,
            0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,2,0,0,
            0,0,0,0,0,0,0,1,0,
            0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,1,
            0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,
        ]
    else:
        raise ValueError("Unknown level")
    # fmt: on

    if args.algo == "ac3":
        solver = sudoku
    elif args.algo == "custom":
        solver = my_sudoku
    else:
        raise ValueError("Unknown algorithm type")

    times = []
    for i in range(args.trials):
        test_board = board[:]  # Ensure original board is not modified
        start = time.perf_counter()
        solution, recursions = solver(test_board)
        end = time.perf_counter()
        times.append(end - start)
        if solution and not check_solution(solution, board):
            print(solution)
            raise ValueError("Invalid solution")

        if solution:
            print(f"Trial {i} solved with {recursions} recursions")
            print(solution)
        else:
            print(f"Trial {i} not solved with {recursions} recursions")

    print(
        f"Minimum time {min(times)}s, Average time {sum(times) / args.trials}s (over {args.trials} trials)"
    )
