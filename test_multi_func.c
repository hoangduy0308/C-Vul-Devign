#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Vulnerable function - buffer overflow
void vulnerable_copy(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // Buffer overflow!
    printf("Copied: %s\n", buffer);
}

// Safe function - no issues
int safe_add(int a, int b) {
    return a + b;
}

// Vulnerable function - unchecked malloc
char* vulnerable_alloc(size_t size) {
    char *ptr = malloc(size);
    // Missing NULL check!
    memset(ptr, 0, size);
    return ptr;
}

// Vulnerable function - use after free potential
void vulnerable_free(char *ptr) {
    free(ptr);
    // ptr is now dangling, should set to NULL
}

// Safe function with proper checks
char* safe_strdup(const char *str) {
    if (str == NULL) {
        return NULL;
    }
    
    size_t len = strlen(str) + 1;
    char *copy = malloc(len);
    
    if (copy == NULL) {
        return NULL;
    }
    
    memcpy(copy, str, len);
    return copy;
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input>\n", argv[0]);
        return 1;
    }
    
    vulnerable_copy(argv[1]);
    
    char *ptr = vulnerable_alloc(100);
    vulnerable_free(ptr);
    
    int sum = safe_add(10, 20);
    printf("Sum: %d\n", sum);
    
    return 0;
}
