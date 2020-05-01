from numpy import zeros

MAPPING_STRING = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "i": 8,
    "j": 9,
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "q": 16,
    "r": 17,
    "s": 18,
    "t": 19,
    "u": 20,
    "v": 21,
    "w": 22,
    "x": 23,
    "y": 24,
    "z": 25,
    "1": 26,
    "2": 27,
    "3": 28,
    "4": 29,
    "5": 30,
    "6": 31,
    "7": 32,
    "8": 33,
    "9": 34,
    "0": 35,
}


# Input: str
# Output: numpy array of corresnponding int
def string_to_array(s):
    string_array = zeros(shape=(36))
    i = 0
    for c in s:
        string_array[MAPPING_STRING.get(c) + (36 * i)] = 1
        i += 1
    return string_array


# Input: numpy array of int
# Output: str of corresnponding chars
def array_to_string(arr):
    string = ""
    for c in arr:
        i = c % 36
        for key, value in MAPPING_STRING.items():
            if i == value:
                string += key
    return string
