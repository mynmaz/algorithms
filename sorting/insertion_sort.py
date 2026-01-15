from typing import List                         # Adds list type hinting for clarity

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
