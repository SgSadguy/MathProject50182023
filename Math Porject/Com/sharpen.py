def base62_decode(encoded):
    base62_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    base62_dict = {char: idx for idx, char in enumerate(base62_chars)}
    
    decoded_value = 0
    power = 0
    
    for char in encoded[::-1]:
        decoded_value += base62_dict[char] * (62 ** power)
        power += 1
    
    return decoded_value

encoded_string = "1Ke0zSzxd539Z1cYhF4ILB80p5XGj0J0MhNrA1zo07UVx"
decoded_value = base62_decode(encoded_string)
print(f"Decoded value: {decoded_value}")