from tokenizer import chinese_tokenizer

def get_token(line):
    'line: str or List[str].'
    token = chinese_tokenizer(line, padding=True, truncation=True)
    return token

