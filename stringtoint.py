def atoi(s):
    """Transform les strings en integers"""
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0") # le code ascii de i et de "0"
    return n
