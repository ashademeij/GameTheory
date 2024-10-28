import itertools
import heapq

#### Problem Summary and Input Handling
# Problem Summary:
# We have n walkers, each with a specific crossing time.
# The bridge can hold only a specified number of walkers at a time.
# The crossing time can be calculated based on the user's choice (e.g., max, min, average).
# A torch must be carried across, so someone has to bring it back after each crossing.
# We need to find the minimum time for all walkers to cross the bridge.


#### Methods

def get_input():
    """
    This method retrieves the user inputs (number of walkers, each walker's crossing time, bridge capacity
    and the rule for calculating group crossing time)
    
    Returns:
    tuple: (walkers, bridge_capacity, crossing_rule_choice)
        - walkers (dict): Walker IDs and their corresponding crossing times
        - bridge_capacity (int): Maximum number of walkers the bridge can hold at once
        - crossing_rule_choice (int): 1 = maximum time, 2 = mininum time, 3 = average time)
    """
    
    # Get number of walkers
    while True:
        num_walkers = int(input("Enter the number of walkers (positive integer): "))
        if num_walkers > 0:
            break
        else:
            print("The number of walkers must be a positive integer. Please try again.")

    # Get crossing times for each walker
    walkers = {}
    for i in range(1, num_walkers + 1):
        while True:
            crossing_time = int(input(f"Enter the crossing time for walker {i} (positive integer min): "))
            if crossing_time > 0:
                walkers[i] = crossing_time
                break
            else:
                print("Crossing time must be a positive integer. Please try again.")

    # Get bridge capacity
    while True:
        bridge_capacity = int(input("Enter the maximum number of people the bridge can hold at once (positive integer): "))
        if bridge_capacity > 0:
            break
        else:
            print("Bridge capacity must be a positive integer. Please try again.")

    # Get user's choice for crossing rule
    while True:
        print("Choose the crossing rule (how the crossing time is calculated):")
        print("1. Maximum time (slowest walker determines the time)")
        print("2. Minimum time (fastest walker determines the time)")
        print("3. Average time (average of all walkers in the group)")
        crossing_rule_choice = int(input("Enter the number of your choice (1/2/3): "))
        
        if crossing_rule_choice in [1, 2, 3]:
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Return all user inputs as a tuple
    return walkers, bridge_capacity, crossing_rule_choice


def calculate_crossing_time(group, walkers, crossing_rule_choice):

    """ This method function calculates the crossing time for a group based on the chosen rule

    Parameters:
    group (list): list of walker IDs 
    walkers (dict): Walker IDs and their corresponding crossing times
    crossing_rule_choice (int): 1 = maximum time, 2 = mininum time, 3 = average time)

    Returns:
    float: crossing time for the walkers based on the selected rule

   """

    if crossing_rule_choice == 1:  # Maximum time
        return max(walkers[m] for m in group)
    elif crossing_rule_choice == 2:  # Minimum time
        return min(walkers[m] for m in group)
    elif crossing_rule_choice == 3:  # Average time
        return sum(walkers[m] for m in group) / len(group)
    

def possible_forward_moves(start_side):

    """ This method function returns a list with all possible groups that can move from the left to the right side of the bridge w.r.t to the bridge capacity

    Parameters:
    start_side (list): list of walkers on the left side of the bridge

    Returns:
    list: list of tuples representing possible groups that can move foward

   """
    
    # Generate groups of walkers w.r.t the bridge capacity and available walkers
    max_group_size = min(bridge_capacity, len(start_side)) 
    if max_group_size >= 2:  # We need at least 2 walkers to cross forward
        return list(itertools.combinations(start_side, max_group_size))  
    else:
        return []


def possible_backward_moves(end_side):

    """ This method function returns a list of single walkers that can move from the right to the left side of the bridge

    Parameters:
    end_side (list): list of walkers on the right side of the bridge

    Returns:
    list: list of single walkers who are currently on the right side of the bridge

   """

    if len(end_side) >= 1:  # ensures that 1 walker on the end side to bring the torch back.
        return [(walker,) for walker in end_side] 
    else:
        return []


def generate_graph(walkers, crossing_rule_choice):

    """ This method generates a graph with its nodes standing for all possible states and and the edges which 
    represent all possible transitions of the bridge crossing problem.

    Parameters:
    walkers (dict): Dictionary of walker IDs and their crossing times
    crossing_rule_choice (int): 1 = maximum time, 2 = mininum time, 3 = average time)

    Returns:
    dict: Which represents a graph where:
        - Key: represents a state as a tuple (start_side, end_side, torch_position)
        - Value: list of transitions possible from the state represented as tuples (new_state, crossing_time, move)
    """
        
    graph = {} 
    queue = [initial_state] 
    visited = set() 
    
    while queue:
        current_state = queue.pop(0)  
        current_start, current_end, current_torch = current_state  
        
        if current_state in visited:
            continue  # Skip if visited
        
        visited.add(current_state)  # Mark the current state as visited
        graph[current_state] = []  # Initialize the list of possible transitions for the current state
        
        # case 1: Moving Forward (Group of Walkers Cross + Torch is on start side)
        # Get all possible groups of walkers to move forward
        if current_torch == 'start': 
            forward_moves = possible_forward_moves(current_start) 
            
            for move in forward_moves:
                new_start = set(current_start) - set(move)  # Remove the current walkers from the start side
                new_end = set(current_end) | set(move)  # Add the current walkers to the end side
                new_state = (tuple(sorted(new_start)), tuple(sorted(new_end)), 'end')  
                crossing_time = calculate_crossing_time(move, walkers, crossing_rule_choice)  
                graph[current_state].append((new_state, crossing_time, move)) 
                queue.append(new_state)  
        

        # case 2: Moving Backward (One Walker Returns + Torch is on end side)
        # Get all possible walkers to move backward
        elif current_torch == 'end': 
            backward_moves = possible_backward_moves(current_end) 
            
            # For each possible backward move, update to the new state, calculate the crossing time, and add the new state to the graph
            for move in backward_moves:
                new_start = set(current_start) | set(move)  # Add the returning walker to the start side
                new_end = set(current_end) - set(move)  # Remove the returning walker from the end side
                new_state = (tuple(sorted(new_start)), tuple(sorted(new_end)), 'start')  
                crossing_time = walkers[move[0]] 
                graph[current_state].append((new_state, crossing_time, move)) 
                queue.append(new_state)  

    return graph

def dijkstra(graph, start, goal):

    """ This method employs the Dijkstra's algorithm to find the the shortest paths from the start state to the goal state 

    Parameters:
    graph (dict): graph where keys are states (start_side, end_side, torch_position) & values are lists of transitions (new_state, crossing_time, move)
    start (tuple): initial state
    goal (tuple): target state

    Returns:
    list: A list of tuples (path, cost) representing all possible solutions
    """

    pq = [(0, start, [])]  # Priority queue of (cost, state, path), initially starts with the start state and 0 cost
    solutions = []  

    while pq:
        current_cost, current_state, current_path = heapq.heappop(pq)  # Pop the state with the lowest cost
        
        if current_state == goal:  # If we reach the goal state, store the solution
            solutions.append((current_path, current_cost))
            continue
        
        # Explore all neighbors of the current state
        for neighbor, weight, move in graph[current_state]:  
            new_cost = current_cost + weight  # Calculate the new cost (current cost + time to reach the neighbor)
            heapq.heappush(pq, (new_cost, neighbor, current_path + [(move, weight, current_state, neighbor)]))  # Push the neighbor onto the priority queue with the new cost and updated path
    
    return solutions


#### Main method
if __name__ == "__main__":

    # Get the user input
    walkers, bridge_capacity, crossing_rule_choice = get_input()

    # Iniatilize the bridge crossing problem based on the input 
    start_side = list(walkers.keys())  
    end_side = []  
    torch_side = "start"  
    initial_state = (tuple(start_side), tuple(end_side), torch_side)
    goal_state = (tuple([]), tuple(start_side), "end")

    # Generate the graph + Run Dijkstra's Algorithm
    graph = generate_graph(walkers, crossing_rule_choice)  
    solutions = dijkstra(graph, initial_state, goal_state)  

    # Output/print the solutions
    print(f"Best Solutions:")


    # Find the solution with the minimum total time
    min_time = min(total_time for _, total_time in solutions)

    # Filter only the best solutions (those with the minimum total time)
    best_solutions = [(path, total_time) for path, total_time in solutions if total_time == min_time]

    # Enumerate through only the best solutions
    for idx, (path, total_time) in enumerate(best_solutions, 1):  
        moves = []
        for move, time, prev_state, next_state in path:  # For each step in the solution path
            if len(move) > 1:
                moves.append(f"{'+'.join(map(str, move))}")  # For group moves, display them as a joined string
            else:
                moves.append(f"{move[0]} back")  # For backward moves (one walker returns), display who returned
        
        # Show the full state transitions (who is on each side and the torch's location)
        for move, _, prev_state, next_state in path:
            print(f"{prev_state} -> {next_state}")  
        
        # transforming the solution into an easier readable format
        solution_str = ', '.join(moves) 
        print(f"Readable Moves: ({solution_str}), Total Time: {total_time} min")





