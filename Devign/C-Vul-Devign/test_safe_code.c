// Safe C code example - no vulnerabilities
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void safe_function(const char *user_input, size_t input_len) {
    char buffer[64];
    
    // Safe: bounds checking with strncpy
    if (input_len > 0 && input_len < sizeof(buffer)) {
        strncpy(buffer, user_input, sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
    }
    
    // Safe: using snprintf instead of sprintf
    char output[128];
    snprintf(output, sizeof(output), "Input: %s", buffer);
    printf("%s\n", output);
    
    // Safe: NULL check after malloc
    char *ptr = malloc(100);
    if (ptr == NULL) {
        return;
    }
    memset(ptr, 0, 100);
    free(ptr);
    ptr = NULL;  // Safe: set to NULL after free
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        safe_function(argv[1], strlen(argv[1]));
    }
    return 0;
}
