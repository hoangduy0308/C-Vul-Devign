#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerable_function(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // Buffer overflow vulnerability
    printf("%s\n", buffer);
    
    char *ptr = malloc(100);
    memcpy(ptr, input, 100);
    
    free(ptr);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    }
    return 0;
}
