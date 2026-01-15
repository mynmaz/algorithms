from typing import List                         # Adds list type hinting for clarity

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
