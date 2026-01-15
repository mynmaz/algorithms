#!/usr/bin/env python3  # Use Python 3 interpreter

"""
main.py
Author: Eddie Sweeney

Purpose:
Menu-based suite of sorting algorithms with detailed timing, comparisons, and CSV output.
Includes:
- Choice of specific or all algorithms
- Benchmarking best/average/worst cases
- Handles very large inputs 
- Saves benchmark results & Big-O CSV tables
- Custom or0iginal algorithm: eddiesalgorithm_sort 
"""

# =======================
# Imports
# =======================
import csv                # Used to read/write CSV files for benchmark data
import random             # Used to generate random integers for sorting and pivots if needed
import time               # Used to measure performance timing of algorithms
import heapq              # Provides heap functions for Heap Sort
from typing import List   # Adds list type hinting for clarity
import sys                # Gives system-level access (like recursion limit)

sys.setrecursionlimit(20000)  # Raise recursion depth limit for large recursive sorts (merge/quick, if used)

# =======================
# Sorting Algorithms 
# =======================

def bubble_sort(a: List[int]) -> List[int]:  # Bubble Sort function 
    arr = list(a)                            # Copy input list to avoid mutating caller data
    n = len(arr)                             # Cache length for reuse
    for i in range(n):                       # Outer pass runs n times (worst-case)
        swapped = False                      # Track whether any swap occurs in this pass
        for j in range(0, n - i - 1):        # Inner pass compares adjacent pairs up to unsorted boundary
            if arr[j] > arr[j + 1]:          # If current element is greater than next element
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Swap the elements
                swapped = True               # Mark that a swap occurred
        if not swapped:                      # If no swaps happened in this pass
            break                            # Array is sorted; exit early optimization
    return arr                               # Return the sorted array


def insertion_sort(a: List[int]) -> List[int]:  # Define Insertion Sort function
    arr = list(a)                               # Copy input list
    for i in range(1, len(arr)):                # Iterate from the second element to the end
        key = arr[i]                            # Value to insert into the sorted prefix
        j = i - 1                               # Start comparing from the previous element
        while j >= 0 and arr[j] > key:          # While elements to the left are greater than key
            arr[j + 1] = arr[j]                 # Shift larger element one position to the right
            j -= 1                              # Move leftward
        arr[j + 1] = key                        # Insert key into the correct position
    return arr                                  # Return sorted array


def selection_sort(a: List[int]) -> List[int]:  # Define Selection Sort function
    arr = list(a)                               # Copy input
    n = len(arr)                                # Cache length
    for i in range(n):                          # Position where we will place the next minimum
        min_idx = i                             # Assume current index holds the minimum
        for j in range(i + 1, n):               # Scan the remainder of the array
            if arr[j] < arr[min_idx]:           # If a smaller element is found
                min_idx = j                     # Update index of minimum
        arr[i], arr[min_idx] = arr[min_idx], arr[i]  # Swap min element into correct position
    return arr                                  # Return sorted array


def merge_sort(a: List[int]) -> List[int]:      # Define Merge Sort (divide and conquer)
    arr = list(a)                               # Copy input list
    if len(arr) <= 1:                           # Base case: 0 or 1 element is already sorted
        return arr                              # Return as-is
    mid = len(arr) // 2                         # Compute midpoint
    left = merge_sort(arr[:mid])                # Recursively sort left half
    right = merge_sort(arr[mid:])               # Recursively sort right half
    result = []                                 # Prepare output buffer
    i = j = 0                                   # Pointers for left and right lists
    while i < len(left) and j < len(right):     # Merge until one side is exhausted
        if left[i] < right[j]:                  # If left item is smaller
            result.append(left[i])              # Append left item
            i += 1                              # Advance left pointer
        else:                                   # Otherwise right item is smaller or equal
            result.append(right[j])             # Append right item
            j += 1                              # Advance right pointer
    result.extend(left[i:])                     # Append remaining left items
    result.extend(right[j:])                    # Append remaining right items
    return result                               # Return merged sorted list


def quick_sort(a: List[int]) -> List[int]:      # Define Quick Sort (functional style)
    arr = list(a)                               # Copy input
    if len(arr) <= 1:                           # Base case: already sorted
        return arr                              # Return as-is
    pivot = random.choice(arr)                  # Choose a random pivot to reduce worst-case risk on patterns
    less = [x for x in arr if x < pivot]        # Elements less than pivot
    equal = [x for x in arr if x == pivot]      # Elements equal to pivot (to handle duplicates)
    greater = [x for x in arr if x > pivot]     # Elements greater than pivot
    return quick_sort(less) + equal + quick_sort(greater)  # Recursively sort partitions and concatenate


def counting_sort(a: List[int]) -> List[int]:   # Define Counting Sort (non-comparative)
    if not a:                                   # If input is empty
        return []                               # Return empty list
    if min(a) < 0:                              # Validate domain (non-negative integers only)
        raise ValueError("Counting sort requires non-negative integers.")  #  precondition
    max_val = max(a)                            # Determine maximum value to size the count array
    count = [0] * (max_val + 1)                 # Initialize frequency counts for each value
    for num in a:                               # Iterate over input
        count[num] += 1                         # Increment frequency for that value
    result = []                                 # Prepare output list
    for val, freq in enumerate(count):          # For each possible value and its frequency
        result.extend([val] * freq)             # Append 'val' repeated 'freq' times
    return result                               # Return sorted result


def bucket_sort(a: List[int]) -> List[int]:     # Define Bucket Sort (simple numeric range version)
    if len(a) == 0:                             # Handle empty input
        return a                                # Return as-is
    min_val, max_val = min(a), max(a)           # Determine range of data
    bucket_count = len(a)                       # Use one bucket per element (simple strategy)
    bucket_range = (max_val - min_val + 1) / bucket_count  # Compute bucket width
    buckets = [[] for _ in range(bucket_count)] # Create list of empty buckets
    for num in a:                               # Distribute each number into a bucket
        idx = int((num - min_val) / bucket_range)  # Compute bucket index
        if idx == bucket_count:                 # If index falls on the upper edge
            idx -= 1                            # Clamp to last bucket
        buckets[idx].append(num)                # Append number into that bucket
    result = []                                 # Prepare output
    for b in buckets:                           # For each bucket
        result.extend(sorted(b))                # Sort the bucket and append
    return result                               # Return concatenated sorted output


def counting_sort_for_radix(a: List[int], exp: int):  # Helper for Radix Sort on a given digit exp
    n = len(a)                                  # Cache length
    output = [0] * n                            # Output array for stable distribution
    count = [0] * 10                            # Digit frequency (0-9)
    for i in range(n):                          # Count digit occurrences at current exponent
        index = (a[i] // exp) % 10              # Extract digit
        count[index] += 1                       # Increment its count
    for i in range(1, 10):                      # Compute cumulative counts
        count[i] += count[i - 1]                # Accumulate previous counts
    for i in range(n - 1, -1, -1):              # Build output in reverse for stability
        index = (a[i] // exp) % 10              # Extract digit again
        output[count[index] - 1] = a[i]         # Place element at correct position
        count[index] -= 1                       # Decrement position pointer
    for i in range(n):                          # Copy back to input array
        a[i] = output[i]                        # Overwrite with stable sorted by current digit


def radix_sort(a: List[int]) -> List[int]:      # Define Radix Sort (LSD)
    arr = list(a)                               # Copy input
    if not arr:                                 # Handle empty list
        return arr                              # Return empty
    if min(arr) < 0:                            # Validate domain for this implementation
        raise ValueError("Radix Sort requires non-negative integers.")  #  non-negative input
    max_val = max(arr)                          # Find maximum to know number of digits
    exp = 1                                     # Start with ones place
    while max_val // exp > 0:                   # While there are still digits to process
        counting_sort_for_radix(arr, exp)       # Stable sort by current digit
        exp *= 10                               # Move to next digit place
    return arr                                  # Return sorted array


def heap_sort(a: List[int]) -> List[int]:       # Define Heap Sort
    arr = list(a)                               # Copy input list
    heapq.heapify(arr)                          # Transform into a min-heap in-place
    return [heapq.heappop(arr) for _ in range(len(arr))]  # Pop elements in ascending order



# =======================
# Tree Sort (Non-Recursive helpers)
# =======================
class TreeNode:                                 # Simple binary search tree node
    def __init__(self, value):                  # Node initializer
        self.value = value                      # Store node's value
        self.left = None                        # Left child pointer
        self.right = None                       # Right child pointer


def _insert(root, value):                       # Iterative insert into BST
    new_node = TreeNode(value)                  # Create a new node
    current = root                              # Start at root
    while True:                                 # Loop until we place the node
        if value < current.value:               # If value should go to the left subtree
            if current.left is None:            # If left spot is empty
                current.left = new_node         # Insert here
                break                           # Done
            current = current.left              # Else descend left
        else:                                   # Otherwise it belongs to the right subtree
            if current.right is None:           # If right spot is empty
                current.right = new_node        # Insert here
                break                           # Done
            current = current.right             # Else descend right
    return root                                  # Return root (unchanged reference)


def _inorder_traversal_iterative(root):         # Iterative in-order traversal of BST
    result = []                                  # Output list for visited values
    stack = []                                   # Stack to simulate recursion
    current = root                               # Start from the root
    while stack or current:                      # Continue while nodes remain
        while current:                           # Reach leftmost node
            stack.append(current)                # Push node onto stack
            current = current.left               # Move left
        current = stack.pop()                    # Pop the next unvisited node
        result.append(current.value)             # Visit the node (append its value)
        current = current.right                  # Move to right subtree
    return result                                 # Return in-order sequence (sorted if BST balanced enough)


def tree_sort(a: List[int]) -> List[int]:        # Tree Sort using iterative helpers
    arr = list(a)                                 # Copy input
    if not arr:                                   # Handle empty input
        return []                                 # Return empty
    root = TreeNode(arr[0])                       # Initialize BST with first element
    for val in arr[1:]:                           # Insert remaining values
        _insert(root, val)                        # Insert iteratively
    return _inorder_traversal_iterative(root)     # Return in-order traversal result

# =======================
# Tim Sort (built-in)
# =======================
def tim_sort(a: List[int]) -> List[int]:         # Wrapper to Python's built-in Timsort
    return sorted(a)      # return sorted array 

# =======================
# ERDS - Eddie's Random Drift Sort (Eddie's custom ( great for sorting) (Algorithm)
# =======================
# Idea : 
# - Repeatedly picks a random pivot index. 
# - The Values on the Left of the pivot that are too big drift to the right side. 
# - The Values on the right of the pivot that are too small drift to the left side. 
# - The pivot here "drifts" through the array until the list stabilizes into sorted order.
# - drift-based randomized sorting process designed for this project.(Remember I can use this custom algorithm for my cronus zen)
# Time Complexity: 
# Best Case: O(n log n) 
# Average Case: O(n^2)
# Worst Case: O(n^3)    (unlucky pivots + bad drift cascades= like falling dominoes)


def eddies_random_drift_sort(a: List[int]) -> List[int]:
    """ERDS: Eddie's Random Drift Sort (custom randomized sorting algorithm)"""

    arr = list(a)                 # Make a full copy so we do not modify the caller's data
    n = len(arr)                  # Cache the length for efficiency

    if n <= 1:                    # Edge case: 0 or 1 element is already sorted
        return arr                # Return as-is

    # ------------------------------------------------------------------
    # Helper function — determines if the array is fully sorted
    # ------------------------------------------------------------------
    def is_sorted(xs):
        # Check every adjacent pair; array sorted only if xs[i] <= xs[i+1]
        return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))

    # ------------------------------------------------------------------
    # Main ERDS Loop
    # Continue running drift passes until array becomes sorted
    # ------------------------------------------------------------------
    while not is_sorted(arr):  # Keep going until sorted 

        # Pick a random pivot index from 0 to n-1
        p = random.randint(0, n - 1)

        pivot = arr[p]            # Save the pivot’s value for comparisons
        changed = False           # Track if this drift pass made any changes

        # --------------------------------------------------------------
        # Sweep across entire array and drift items around pivot
        # --------------------------------------------------------------
        for i in range(n): # Iterate through each index in the array

            # --------------------------------------------
            # Case 1: Element is left of pivot but is too big
            # --------------------------------------------
            if i < p and arr[i] > pivot: # Element left of pivot but larger than pivot

                # Swap arr[i] with arr[p] → drift pivot LEFT
                arr[i], arr[p] = arr[p], arr[i]

                p = i            # Pivot moves to new position i
                pivot = arr[p]   # Update pivot value
                changed = True   # Mark that something changed

            # --------------------------------------------
            # Case 2: Element is right of pivot but is too small
            # --------------------------------------------
            elif i > p and arr[i] < pivot:  # Element right of pivot but smaller than pivot

                # Swap arr[i] with arr[p] → drift pivot right
                arr[i], arr[p] = arr[p], arr[i]

                p = i            # Pivot now located at index i
                pivot = arr[p]   # Update pivot value
                changed = True   # Mark that we altered the array

        # If no changes were made during the entire sweep,
        # the array might still be unsorted due to bad pivot position.
        # ERDS will simply try another random pivot in the next loop iteration.
        # This prevents infinite loops.

    # Once sorted, return the final array
    return arr



# =======================
# Deterministic Machine
# =======================

# Deterministic = no guessing,no branching (For any current state and input, there is only one defined next step.)
#Properties of My deterministic machine: 
# - Finite set of states: {"EVEN", "ODD"}
#- Start state: (zero 1s seen so far)
# - Accepting state(s): {"EVEN"} ( string with an even number of 1s)
# - Alphabet: {'0', '1'}
# - Transition function: (state, symbol) -> next state is single-valued
# (for any (state, symbol) pair there is exactly ONE next state ----> determinstic)

#Transition table for the DFA
DFA_TRANSITIONS ={  # Define the transition function
    "EVEN": { #Currently seen an even number of '1's
        '0': "EVEN", # Reading '0'does not change parity 
        '1': "ODD"   # Reading '1' flips to odd 
    },
    "ODD": { # Currently seen an odd number of '1's
        '0': "ODD", # Reading '0' keeps us in odd 
        '1': "EVEN" # Reading '1' flips back to even 
        }
}
DFA_START_STATE = "EVEN" # Start with sero 1s → even 
DFA_ACCEPT_STATES = {"EVEN"} # Accept strings with an even number of 1s 

def dfa_even_ones_accepts(s: str) -> bool:  # Define DFA simulation function
    """Simulate the deterministic machine (DFA) on a binary string.
    :param s: Input string over alphabet {'0','1'} 
    :return: True if the DFA ends in  an acceptings state (even number of '1's)), False otherwise.
    """
    state = DFA_START_STATE     # Always begin in the start state
    for ch in s:                # Iterate through each character in the input string
        if ch not in ('0', '1'):                #Process each character in the input string 
           raise ValueError("DFA only accepts binary string containing '0' and '1'.") # Invalid character
        
        state = DFA_TRANSITIONS[state][ch]     # Move determinstical,y to the next state
    return state in DFA_ACCEPT_STATES      # Accept if final state is an accepting state 


# =======================
# Shortest Path Problem (Graphs)
# =======================
# So I will implement two deterministic algorithms (note for me: commonly used to find the shortest paths:)
# 1. BFS (Breadth-First Search) 
#     - Works on unweighted graphs 
#     - Determinstic: explores layer-by-layer
#     - Guarantees the path with the fewest edges 
#

# 2. Dijkstra's Algorithm 
#    - Works on weighted graphs with non-negative weights 
#    - Dterministic priority-queue based selection 
#    - Guarantees minimum total cost path 
#
# Both are classical shortest-apth solutions in CS. 

def bfs_shortest_path(graph: dict, start, goal):   # defining BFS shortest path
    """ 
    Compute the shortest path between two nodes in an unweighted graph using Breadth-First Search (BFS). 
    :param graph: adjacency list representation {node: [neighbors]}
    :param start: start node
    :param goal: target node
    :return: list representing the shortest path (or None if)
    """

    from collections import deque  # Import deque for efficient queue operations

    queue = deque([[start]])   # Queue stores full paths (not just nodes)
    visited = set([start])     # Track visited nodes deterministically

    while queue: 
        path = queue.popleft()  # Get next path in queue
        node = path [-1]        # inspect last node in the path 

        if node == goal:        # If we've reached the goal  -> return path 
            return path 
        
        for neighbor in graph.get(node, []):   # Explore neighbors
            if neighbor not in visited:        # If neighbor not yet visited
                visited.add(neighbor)          # Mark as visited
                queue.append(path + [neighbor])    # Build a new path 


    return None      # No path found 

def dijkstra_shortest_path(graph: dict, start, goal):  # defining Dijkstra's algorithm
    """
    Compute the minium-cost path between two nodes using Dijkstra's Algorithm.
    
    :param graph: weighted adjacency list {node: [(neighbor, weight), ...]}
    :param start: starting node 
    :param goal: target node 
    :return: (total_cost, path_list)
    """
    import heapq  # Import heapq for priority queue functionality

    # Min-heap priority queue storing (cost_so_far, node, path_taken)
    pq = [(0, start, [start])]
    visited = set()   # Track visited nodes 
      
    while pq:         # While there are nodes to process
        cost, node, path = heapq.heappop(pq)   # Get node with lowest cost

        if node in visited:  # If already visited, skip
            continue         # Skip already visited nodes
        visited.add(node)    # Mark node as visited
    
        for neighbor, weight in graph.get(node, []):    # Explore neighbors with weights
            if neighbor not in visited:                 # If neighbor not yet visited
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))     # Add new path to priority queue

    return float('inf'),  []   # Goal unreachable 


def shortest_path_demo():  # Shortest Path Demo
    """shows the BFS and Dijkstra's shortest path algorithms on small graphs.  """
    print("\n=== Shortest Path Demo ===")

    #Unweighted graph for BFS 
    graph_unweighted = { 
        'A': ['B', 'C'],
        'B': ['D'], 
        'C': ['D', 'E'], 
        'D': ['F'],
        'E': ['F'],
        'F': []
    }

    print("\n[Unweighted Graph] BFS shortest path A -> F:")  # BFS shortest path
    print("Path:",  bfs_shortest_path(graph_unweighted, 'A', 'F'))   # Print the path found by BFS


    # Weighted graph for  Dijkstra
    graph_weighted = {
        'A': [('B', 2), ('C', 4)],
        'B': [('D', 3)], 
        'C': [('D', 1)], 
        'D': [('E', 5)],
        'E': []

    }

    print("\n[Weighted Graph] Dijkstra shortest path  A --> E:")   # Dijkstra shortest path
    cost, path = dijkstra_shortest_path(graph_weighted, 'A', 'E')  # Get cost and path
    print("Cost:", cost)    #  print the total cost 
    print("Path", path)     # print the path taken
 
def dfa_demo():     # DFA Demo
    """ 
    Small demo function to show how the deterministic machine behaves on a few example inputs. This is optional and can be called from main(). 
    """
    print("\n=== Determinstic Machine (DFA) Demo ===")    # Demo header
    examples = ["", "0","1","10", "11", "101", "1111", "101010"]   # Test strings with various counts of '1's
    for s in examples:            # Iterate through test strings
        try:                      # Try block to catch invalid input
            accepted =dfa_even_ones_accepts(s)           # Run DFA on test string 
            Label = "Accept" if accepted else "REJECT"   # Classify acoording to final state 
            print(f"Input '{s or 'ε'}'--> {Label}")        # Use ε for empty string display 
        except ValueError as e:                            # Catch invalid input errors
            print(f"Input '{s}' -> ERROR: {e}")            # Print error message
 
# =======================
# Gaussian Elimination (Deterministic Linear Algebra Method)
# =======================
# Gaussian Elimination is a deterministic algorithm for solving systems of linear equations.
# steps for eddie: 
# 1. Forward Elimination -> convert matrix A into upper triangular form 
# 2. Back Substitution   -> solve variables from botton to top
#
# Complexity: O(n^3)
#
# This implementation solves: 
# So This implemenetation will solve A = n*n matrix (list of lists)
# b = constants vector (list of size n)
#
#Returns: solution vector x (size n)

def gaussian_elimination(A, b):    # Solve Ax = b using Gaussian Elimination
    """
    Solve Ax = b using classical Gaussian Elimination (without pivoting).

    :param A: coefficient matrix as list of lists (n × n)
    :param b: constants vector (n)
    :return: solution vector x (n)
    """
    n = len(A)     # Number of equations / variables

    # Make deep copies to avoid modifying caller's matrix/vector
    A = [row[:] for row in A]   # Copy of A
    b = b[:]            #  Copy of b

    # === Forward Elimination ===
    for i in range(n):

        # Check for zero pivot (singular system)
        if A[i][i] == 0:
            raise ValueError("Zero pivot encountered — system may be singular.")

        # Eliminate entries below the pivot row i
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]   # How much of row i to subtract

            # Subtract factor * row i from row j
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]

            b[j] -= factor * b[i]

    # === Back Substitution ===
    x = [0] * n  # solution vector

    # Solve from last row upward
    for i in range(n - 1, -1, -1):   # Iterate backwards
        if A[i][i] == 0:    # Check for zero pivot
            raise ValueError("Zero pivot during back substitution — system is singular.")  # Singular check

        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))    # Sum of known variables

        x[i] = (b[i] - sum_ax) / A[i][i]      # Compute variable value 

    return x  # Return solution vector

def gaussian_demo():  # Gaussian Elimination Demo
    """
    Demonstrates Gaussian Elimination on a simple 3×3 linear system.
    Solves:
        2x + 3y -  z =  5
        4x +  y + 5z =  6
        2x + 7y + 2z = 14
    """
    print("\n=== Gaussian Elimination Demo ===")     # Demo header
 
    A = [      # Coefficient matrix
        [2, 3, -1],   # Row 1
        [4, 1,  5],   # Row 2
        [-2, 7, 2]    # Row 3
    ]
    b = [5, 6, 14]    # Constants vector

    try:      # Try block to catch errors
        solution = gaussian_elimination(A, b)  # Solve the system
        print("Solution vector x =", solution) # Print the solution
    except ValueError as e:   # Catch singular system errors
        print("Error:", e)    # Print error message
    




# =======================
# Big-O Table
# =======================
BIGO = {                                          # Mapping of algorithm names to theoretical bounds
    "Bubble Sort":       {"Best": "O(n)", "Average": "O(n^2)", "Worst": "O(n^2)"},  # Bubble Sort entry
    "Insertion Sort":    {"Best": "O(n)", "Average": "O(n^2)", "Worst": "O(n^2)"},  # Insertion Sort entry
    "Selection Sort":    {"Best": "O(n^2)", "Average": "O(n^2)", "Worst": "O(n^2)"}, # Selection Sort entry
    "Merge Sort":        {"Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)"}, # Merge Sort entry
    "Quick Sort":        {"Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n^2)"},  # Quick Sort entry
    "Counting Sort":     {"Best": "O(n + k)", "Average": "O(n + k)", "Worst": "O(n + k)"},  # Counting Sort entry
    "Bucket Sort":       {"Best": "O(n + k)", "Average": "O(n + k)", "Worst": "O(n^2)"},   # Bucket Sort entry
    "Radix Sort":        {"Best": "O(d(n + b))", "Average": "O(d(n + b))", "Worst": "O(d(n + b))"},   # Radix Sort entry
    "Heap Sort":         {"Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)"}, # Heap Sort entry
    "ERDS (eddies_random_drift_sort)": {"Best": "O(n log n)", "Average": "O(n^2)", "Worst": "O(n^3)"},   # Eddie's algorithm
    "Tree Sort":         {"Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n^2)"},  # Tree Sort entry
    "Tim Sort":          {"Best": "O(n)", "Average": "O(n log n)", "Worst": "O(n log n)"},  # Timsort entry
}

# =======================
# Algorithm Registry (menu)
# =======================
ALGOS = {                                                        # User-facing names mapped to callables
    "Bubble Sort": bubble_sort,                                  # Bubble Sort entry
    "Insertion Sort": insertion_sort,                            # Insertion Sort entry
    "Selection Sort": selection_sort,                            # Selection Sort entry
    "Merge Sort": merge_sort,                                    # Merge Sort entry
    "Quick Sort": quick_sort,                                    # Quick Sort entry
    "Counting Sort": counting_sort,                              # Counting Sort entry
    "Bucket Sort": bucket_sort,                                  # Bucket Sort entry
    "Radix Sort": radix_sort,                                    # Radix Sort entry
    "Heap Sort": heap_sort,                                      # Heap Sort entry
    "ERDS (eddies_random_drift_sort)": eddies_random_drift_sort, # Eddie's  algorithm 
    "Tree Sort": tree_sort,                                      # Tree Sort entry
    "Tim Sort": tim_sort,                                        # Timsort entry
}

# =======================
# Utility Functions
# =======================
def generate_numbers(n: int, lo: int = 0, hi: int = 100000) -> List[int]:  # Generate data helper
    rng = random.Random(42)                     # Fixed seed for reproducibility across runs
    return [rng.randint(lo, hi) for _ in range(n)]  # Produce list of n random integers in [lo, hi]


def prepare_case(data: List[int], case: str) -> List[int]:  # Arrange data for best/avg/worst
    if case == "best":                           # If benchmarking best case
        return sorted(data)                      # Return ascending-sorted copy
    elif case == "worst":                        # If benchmarking worst case
        return sorted(data, reverse=True)        # Return descending-sorted copy
    return list(data)                            # Otherwise average case → original random order


def time_one_run(fn, data: List[int]) -> float:  # Time a single function call on given data
    t0 = time.perf_counter()                     # Capture start time with high-resolution timer
    fn(list(data))                               # Call function with a copy to avoid side-effects
    return time.perf_counter() - t0              # Return elapsed seconds as float


def run_and_time(name: str, data: List[int], case: str) -> float:  # Orchestrate one benchmark
    arranged = prepare_case(data, case)           # Prepare data according to selected scenario
    fn = ALGOS[name]                              # Lookup algorithm by name
    return time_one_run(fn, arranged)             # Time the algorithm and return seconds


def write_bigO_csv(path="bigO_table.csv"):        # Persist the Big-O table to CSV
    with open(path, "w", newline="") as f:        # Open file for writing in text mode
        w = csv.writer(f)                         # Create CSV writer
        w.writerow(["Algorithm", "Best", "Average", "Worst"])  # Write header row
        for name, b in BIGO.items():              # Iterate through complexity entries
            w.writerow([name, b["Best"], b["Average"], b["Worst"]])  # Write each algorithm row

# =======================
# Main Function (interactive menu with sane defaults)
# =======================
def main():                                       # Entry point for program
    print("\n=== Eddie Sweeney's Sorting Suite ===")  # Display suite header

    default_n = 250_000                           # Set default dataset size to 250,000 elements
    n_raw = input(f"Enter how many numbers to generate [default {default_n}]: ").strip()  # Read user size or blank
    n = default_n if n_raw == "" else int(n_raw)  # Use default if blank; otherwise parse int

    direction_raw = input("Sort direction (A = Ascending, D = Descending) [default A]: ").strip().upper()  # Ask dir
    direction = "A" if direction_raw == "" else direction_raw  # Default to Ascending if user hits Enter

    data = generate_numbers(n)                    # Generate the random dataset of chosen size

    if direction == "D":                          # If user wanted descending prep
        data.sort(reverse=True)                   # Pre-sort descending (for consistency across runs)
    else:                                         # Otherwise
        data.sort()                               # Pre-sort ascending

    print("\nChoose which sorting algorithm to run:")  # Present algorithm menu to user
    algo_names = list(ALGOS.keys())               # Get the registered algorithm names
    for i, name in enumerate(algo_names, 1):      # Enumerate for 1-based display
        print(f"{i}. {name}")                     # Print the index and name
    print(f"{len(algo_names)+1}. Run ALL algorithms")  # Offer an option to run all

    choice = input(f"\nEnter a number (1-{len(algo_names)} or {len(algo_names)+1} for all) [default {len(algo_names)}]: ").strip()  # Read selection
    if choice == "":                               # If user hits Enter
        choice = str(len(algo_names))              # Default to the last single algorithm (Tim Sort)

    print("\nChoose which scenario to test:")      # Ask which case to benchmark
    print("1. Best Case (sorted data)")            # Option 1: best case
    print("2. Average Case (random data)")         # Option 2: average case
    print("3. Worst Case (reverse-sorted data)")   # Option 3: worst case
    print("4. All Three Cases")                    # Option 4: benchmark all cases
    case_choice_raw = input("Enter 1-4 [default 2]: ").strip()  # Read case selection
    case_choice = "2" if case_choice_raw == "" else case_choice_raw  # Default to Average if blank

    case_map = {"1": "best", "2": "avg", "3": "worst", "4": "all"}  # Map numeric menu to case tokens
    case = case_map.get(case_choice, "avg")       # Use 'avg' if invalid option supplied

    if choice == str(len(algo_names)+1):          # If user chose to run all algorithms
        algos_to_run = algo_names                  # Use the entire list
    else:                                          # Otherwise run a single algorithm
        try:                                       # Attempt to parse user's choice
            index = int(choice) - 1                # Convert 1-based to 0-based index
            algos_to_run = [algo_names[index]]     # Pick the selected algorithm
        except (ValueError, IndexError):           # If input invalid or out of range
            print("Invalid choice, exiting.")      # Notify error
            return                                 # Abort run

    cases = ["best", "avg", "worst"] if case == "all" else [case]  # Determine scenarios to benchmark
    rows = []                                      # Prepare container for CSV rows

    for name in algos_to_run:                      # Iterate over the algorithms to run
        if name == "Tree Sort" and len(data) > 20000:  # Skip Tree Sort on very large n to avoid deep trees
            print(f"\n[!] Skipping {name} for {len(data)} items (too large).")  # Inform user of skip
            continue                               # Move to next algorithm
        print(f"\n[i] Running {name} on n={len(data)}...")  # Status line before timing
        results = {}                                # Dictionary to store timings per case
        for c in cases:                             # For each requested case
            t = run_and_time(name, data, c)         # Measure runtime in seconds
            results[c] = t                          # Store result under case key
            print(f"  {c} case: {t*1000:.2f} ms")   # Print result in milliseconds for readability
        rows.append([                               # Build CSV row for this algorithm
            name, len(data),                        # Algorithm name and dataset size
            results.get("best"),                    # Best case seconds or None
            results.get("avg"),                     # Average case seconds or None
            results.get("worst"),                   # Worst case seconds or None
            "OK"                                    # Status flag
        ])

    with open("benchmark_results_wide.csv", "w", newline="") as f:  # Open CSV file for benchmark results
        w = csv.writer(f)                         # Create writer
        w.writerow(["Algorithm", "n", "Best (s)", "Average (s)", "Worst (s)", "Status"])  # Header
        w.writerows(rows)                         # Write all gathered rows
    print("\n[i] Saved benchmark_results_wide.csv")  # Confirm save

    write_bigO_csv()                               # Persist Big-O summary table to CSV
    print("[i] Saved bigO_table.csv")              # Confirm save

    print("\n=== Big-O Complexity Table ===")      # Print readable Big-O table to console
    for name, b in BIGO.items():                   # Iterate through mapping
        print(f"{name:20s} | Best: {b['Best']:>12s} | Avg: {b['Average']:>12s} | Worst: {b['Worst']:>12s}")  # Nicely aligned

    print("\nAll selected algorithms have finished running successfully!\n")  # Final status message

    # run to see if the deterministic machine (DFA) demo works or nah 
    run_dfa = input("Run determinstic machine (DFA) demo? (Y/N): ").strip().upper()
    if run_dfa == "Y": 
        dfa_demo()
        
    # menu demo for shortest path demo
    run_sp = input("Run shortest path demo? (Y/N): ").strip().upper()
    if run_sp == "Y": 
        shortest_path_demo()

     # Ask user if they want to display the Big-O growth graph
    run_graph = input("Show Big-O growth line graph? (Y/N): ").strip().upper()
    if run_graph == "Y":
        show_bigO_graph()

    # menu demo for gaussian elimination demo
    run_gauss = input("Run Gaussian Elimination demo? (Y/N): ").strip().upper()
    if run_gauss == "Y": 
        gaussian_demo()
# =======================
# Big-O Line Graph (Runtime Visualization)
# =======================
# This function generates a lin graph visual comparing common Big-O growth rates
# Helps me visualized how fast algorithms grow as input size increases
# The graph includes:
#    O(n), O(n log n), O(n^2), O(n^3)
# These represent the growth of algorithms included in this project.
def show_bigO_graph(): 
    import numpy as np  # Import here so it only loads when the graph is requested 
    import matplotlib.pyplot as plt  # Used to generate the runtime graph
    n = np.linspace(1, 1000, 1000)  # Generate 1000 points between 1 and 1000 (smooth curve)

    curves = {
        "O(n)": n,                                       # Linear growth
        "O(n log n)": n * np.log2(n),                    # Log-linear growth
        "O(n^2)": n**2,                                  # Quadratic growth
        "O(n^3)": n**3                                   # Cubic growth
    }

    plt.figure(figsize=(10, 6))      # Create a larger figure window for clarity

    for label, y in curves.items():  # Plot each curve on the graph
        plt.plot(n, y, label=label)

    plt.ylim(0, 1e7)                 # Clamp y-axis for visibility (otherwise n^3 dominates)
    plt.xlabel("n (input size)")     # X-axis label
    plt.ylabel("Growth Rate f(n)")   # Y-axis label
    plt.title("Big-O Growth Comparison (Line Graph)")  # Graph title
    plt.legend()                     # Show legend describing each curve
    plt.tight_layout()               # Reduce empty space around the graph
    plt.show()                       # Display the actual graph window




# =======================
# Run Program
# =======================
if __name__ == "__main__":                        # Only run when executed as a script
    main()                                        # Invoke main entry point
