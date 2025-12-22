import json

with open('Dataset/devign_slice_v2/vocab.json') as f:
    vocab = json.load(f)

print('=== FULL VOCAB (116 tokens) ===')
for tok, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print(f'{idx:3d}: {tok}')

print('\n=== CHECK FOR VAR tokens ===')
var_tokens = [t for t in vocab.keys() if 'VAR' in t.upper() or 'ID' in t]
print(f'VAR/ID related tokens: {var_tokens}')

print('\n=== TOKEN CATEGORIES ===')
special = ['PAD', 'UNK', 'BOS', 'EOS', 'SEP']
normalized = ['ID', 'FUNC', 'NUM', 'STR', 'CHAR', 'FLOAT']
keywords = ['if', 'else', 'for', 'while', 'return', 'int', 'char', 'void', 'struct', 
            'switch', 'case', 'break', 'continue', 'goto', 'default', 'do',
            'static', 'const', 'unsigned', 'signed', 'long', 'short', 'double', 'float',
            'sizeof', 'typedef', 'enum', 'union', 'extern', 'register', 'volatile', 
            'inline', 'restrict', 'NULL', 'true', 'false']

dangerous = ['malloc', 'free', 'calloc', 'realloc', 'alloca',
             'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp', 'strlen',
             'memcpy', 'memmove', 'memset', 'memcmp', 'memchr',
             'sprintf', 'snprintf', 'printf', 'fprintf', 'scanf', 'sscanf', 'fscanf',
             'gets', 'fgets', 'puts', 'fputs',
             'read', 'write', 'open', 'close', 'fopen', 'fclose', 'fread', 'fwrite',
             'vsnprintf']

print(f'Special tokens in vocab: {[t for t in special if t in vocab]}')
print(f'Normalized tokens in vocab: {[t for t in normalized if t in vocab]}')
print(f'Keywords in vocab: {[t for t in keywords if t in vocab]}')
print(f'Dangerous APIs in vocab: {[t for t in dangerous if t in vocab]}')

# Operators/punctuation
operators = [t for t in vocab.keys() if t not in special + normalized + keywords + dangerous]
print(f'\nOperators/Punctuation: {operators}')
