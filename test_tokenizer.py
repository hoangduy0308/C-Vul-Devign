import re

# Copy tokenizer logic directly to avoid import issues
DANGEROUS_APIS = {
    'alloca', 'calloc', 'close', 'fclose', 'fgetc', 'fgets', 'fopen',
    'fprintf', 'fputs', 'fread', 'free', 'fscanf', 'fwrite', 'getc',
    'getchar', 'gets', 'malloc', 'memchr', 'memcmp', 'memcpy', 'memmove',
    'memset', 'open', 'printf', 'puts', 'read', 'realloc', 'scanf',
    'snprintf', 'sprintf', 'sscanf', 'strcat', 'strcmp', 'strcpy',
    'strlen', 'strncat', 'strncmp', 'strncpy', 'vprintf', 'vsnprintf',
    'vsprintf', 'write'
}

C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
    'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
    'unsigned', 'void', 'volatile', 'while', '_Bool', '_Complex', '_Imaginary',
    'NULL', 'true', 'false', 'nullptr'
}

TOKEN_PATTERNS = [
    ('STRING', r'"(?:[^"\\]|\\.)*"'),
    ('CHAR', r"'(?:[^'\\]|\\.)*'"),
    ('COMMENT_MULTI', r'/\*[\s\S]*?\*/'),
    ('COMMENT_SINGLE', r'//[^\n]*'),
    ('HEX', r'0[xX][0-9a-fA-F]+[uUlL]*'),
    ('FLOAT', r'\d+\.\d*[fFlL]?|\.\d+[fFlL]?|\d+[eE][+-]?\d+[fFlL]?'),
    ('NUMBER', r'\d+[uUlL]*'),
    ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('OPERATOR', r'->|<<|>>|<=|>=|==|!=|&&|\|\||[+\-*/%&|^~<>=!]'),
    ('PUNCTUATION', r'[(){}\[\];,.:?#]'),
    ('WHITESPACE', r'\s+'),
]

COMBINED_PATTERN = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_PATTERNS)
TOKEN_REGEX = re.compile(COMBINED_PATTERN)

def tokenize(code):
    tokens = []
    raw_tokens = []
    for match in TOKEN_REGEX.finditer(code):
        token_type = match.lastgroup
        value = match.group()
        raw_tokens.append((value, token_type))
    
    for i, (value, token_type) in enumerate(raw_tokens):
        if token_type in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
            continue
        if token_type == 'STRING':
            tokens.append('STR')
        elif token_type == 'CHAR':
            tokens.append('CHAR')
        elif token_type in ('NUMBER', 'HEX'):
            tokens.append('NUM')
        elif token_type == 'FLOAT':
            tokens.append('FLOAT')
        elif token_type == 'IDENTIFIER':
            # Normalize identifier
            if value in C_KEYWORDS:
                tokens.append(value)
            elif value in DANGEROUS_APIS:
                tokens.append(value)
            else:
                # Check if followed by '(' -> FUNC
                next_idx = i + 1
                while next_idx < len(raw_tokens):
                    next_val, next_type = raw_tokens[next_idx]
                    if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                        break
                    next_idx += 1
                if next_idx < len(raw_tokens) and raw_tokens[next_idx][0] == '(':
                    tokens.append('FUNC')
                else:
                    tokens.append('ID')  # <-- ALL variables become ID
        elif token_type in ('OPERATOR', 'PUNCTUATION'):
            tokens.append(value)
    return tokens

# Test code example
code = '''
void process_data(char *buffer, int size) {
    char *ptr = malloc(size);
    if (ptr == NULL) {
        return;
    }
    strcpy(ptr, buffer);
    int len = strlen(ptr);
    memcpy(dest, ptr, len);
    free(ptr);
}
'''

print('=== ORIGINAL CODE ===')
print(code)

print('=== TOKENIZED ===')
tokens = tokenize(code)
print(' '.join(tokens))

print('\n=== TOKEN MAPPING ===')
print('buffer     -> ID')
print('size       -> ID') 
print('ptr        -> ID')
print('dest       -> ID')
print('len        -> ID')
print('process_data -> FUNC (followed by "(")') 
print('malloc     -> malloc (DANGEROUS API)')
print('strcpy     -> strcpy (DANGEROUS API)')
print('strlen     -> strlen (DANGEROUS API)')
print('memcpy     -> memcpy (DANGEROUS API)')
print('free       -> free (DANGEROUS API)')
print('NULL       -> NULL (C KEYWORD)')
print('1024       -> NUM')
print('"string"   -> STR')
