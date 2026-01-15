
from typing import List #Adds list type hiniting for clairty 

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
