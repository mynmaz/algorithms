from typing import List                         # Adds list type hinting for clarity

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
