def is_unique(x):
    s = set(x)
    if(len(x) == len(s)):
        return True
    else: 
        return False

print(is_unique([1,2,3,1,5,6]))