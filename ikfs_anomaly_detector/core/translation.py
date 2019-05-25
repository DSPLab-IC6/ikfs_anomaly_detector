def translate(word: str) -> str:
    return {
        'expected one argument': 'Ожидается значение',
        'expected at most one argument': 'Ожидается не более одного значения',
        'expected at least one argument': 'Ожидается минимум одно значение',
        'invalid choice: %(value)r (choose from %(choices)s)': 'Неверное значение %(value)r (выберите из %(choices)s)',
        'optional arguments': 'Непозиционные аргументы',
        'positional arguments': 'Позиционные аргументы',
        'show this help message and exit': 'Информация о программе',
        'the following arguments are required: %s': 'Следующие аргументы обязательны: %s',
        'usage: ': 'Использование: ',
    }.get(word, word)
