/**
 * Safe C code examples - No vulnerabilities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Safe: Bounds checking on array access
int safe_array_access(int arr[], int size, int index) {
    if (index < 0 || index >= size) {
        return -1;
    }
    return arr[index];
}

// Safe: NULL check before pointer dereference
void safe_pointer_use(int *ptr) {
    if (ptr == NULL) {
        return;
    }
    *ptr = 42;
}

// Safe: Using strncpy with proper null termination
void safe_string_copy(char *dest, const char *src, size_t dest_size) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return;
    }
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

// Safe: Checking malloc return value
int *safe_malloc(size_t count) {
    if (count == 0 || count > 1000000) {
        return NULL;
    }
    int *ptr = malloc(count * sizeof(int));
    if (ptr == NULL) {
        return NULL;
    }
    memset(ptr, 0, count * sizeof(int));
    return ptr;
}

// Safe: Simple arithmetic with overflow check
int safe_add(int a, int b) {
    if (a > 0 && b > 2147483647 - a) {
        return -1;  // Overflow
    }
    if (a < 0 && b < -2147483648 - a) {
        return -1;  // Underflow
    }
    return a + b;
}

// Safe: Binary search with bounds
int binary_search(int arr[], int size, int target) {
    if (arr == NULL || size <= 0) {
        return -1;
    }
    int left = 0;
    int right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int size = sizeof(arr) / sizeof(arr[0]);
    
    printf("Element at index 2: %d\n", safe_array_access(arr, size, 2));
    printf("Search for 4: index %d\n", binary_search(arr, size, 4));
    printf("Safe add: %d\n", safe_add(10, 20));
    
    return 0;
}
