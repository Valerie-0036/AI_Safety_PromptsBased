import base64
import codecs
def prompt_encoder(prompt, encoding):
    """
    Encode the prompt using the specified encoding.
    If no encoding is specified, a random encoding method will be chosen.
    """
    import random

    # List of all available encoding methods
    if encoding == "atbash":
        return atbash_encode(prompt)
    elif encoding == "caesar":
        return caesar_encode(prompt, 3)  # Example shift value
    elif encoding == "vigenere":
        return vigenere_encode(prompt, "KEY")  # Example key
    elif encoding == "braille":
        return braille_encode(prompt)
    elif encoding == "morse":
        return morse_encode(prompt)
    elif encoding == "pig_latin":
        return pig_latin_encode(prompt)
    elif encoding == "leet":
        return leet_encode(prompt)
    elif encoding == "binary":
        return binary_encode(prompt)
    elif encoding == "hex":
        return hex_encode(prompt)
    elif encoding == "base64":
        return base64_encode(prompt)
    elif encoding == "rot13":
        return rot13_encode(prompt)
    elif encoding == "reverse":
        return reverse_encode(prompt)
    else:
        raise ValueError("Unsupported encoding type.")


def atbash_encode(text):
    """
    Encode the text using the Atbash cipher.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    reversed_alphabet = alphabet[::-1]
    translation_table = str.maketrans(alphabet, reversed_alphabet)
    return text.translate(translation_table)


def caesar_encode(text, shift):
    """
    Encode the text using the Caesar cipher.
    """

    def shift_alphabet(alphabet, shift):
        return alphabet[shift:] + alphabet[:shift]

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shifted_alphabet = shift_alphabet(alphabet, shift)
    translation_table = str.maketrans(alphabet, shifted_alphabet)
    return text.translate(translation_table)


def vigenere_encode(text, key):
    """
    Encode the text using the Vigenère cipher.
    """

    def generate_vigenere_table():
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        table = []
        for i in range(len(alphabet)):
            row = alphabet[i:] + alphabet[:i]
            table.append(row)
        return table

    def vigenere_encrypt(text, key):
        table = generate_vigenere_table()
        encrypted_text = []
        key_length = len(key)
        for i, char in enumerate(text):
            if char.isalpha():
                row = ord(key[i % key_length]) - ord("a")
                col = ord(char) - ord("a")
                encrypted_char = table[row][col]
                encrypted_text.append(encrypted_char)
            else:
                encrypted_text.append(char)
        return "".join(encrypted_text)

    return vigenere_encrypt(text.lower(), key.lower())


def braille_encode(text):
    """
    Encode the text using Braille.
    """
    braille_dict = {
        "a": "⠁",
        "b": "⠃",
        "c": "⠉",
        "d": "⠙",
        "e": "⠑",
        "f": "⠋",
        "g": "⠛",
        "h": "⠓",
        "i": "⠊",
        "j": "⠚",
        "k": "⠅",
        "l": "⠇",
        "m": "⠍",
        "n": "⠝",
        "o": "⠕",
        "p": "⠏",
        "q": "⠟",
        "r": "⠗",
        "s": "⠎",
        "t": "⠞",
        "u": "⠥",
        "v": "⠧",
        "w": "⠺",
        "x": "⠭",
        "y": "⠽",
        "z": "⠵",
    }
    return "".join(braille_dict.get(char, char) for char in text.lower())


def morse_encode(text):
    """
    Encode the text using Morse code.
    """
    morse_dict = {
        "a": ".-",
        "b": "-...",
        "c": "-.-.",
        "d": "-..",
        "e": ".",
        "f": "..-.",
        "g": "--.",
        "h": "....",
        "i": "..",
        "j": ".---",
        "k": "-.-",
        "l": ".-..",
        "m": "--",
        "n": "-.",
        "o": "---",
        "p": ".--.",
        "q": "--.-",
        "r": ".-.",
        "s": "...",
        "t": "-",
        "u": "..-",
        "v": "...-",
        "w": ".--",
        "x": "-..-",
        "y": "-.--",
        "z": "--..",
    }
    return "".join(morse_dict.get(char, char) for char in text.lower())


def pig_latin_encode(text):
    """
    Encode the text using Pig Latin.
    """

    def pig_latin_word(word):
        if word[0] in "aeiou":
            return word + "yay"
        else:
            return word[1:] + word[0] + "ay"

    words = text.split()
    pig_latin_words = [pig_latin_word(word) for word in words]
    return " ".join(pig_latin_words)


def leet_encode(text):
    """
    Encode the text using Leet Speak.
    """
    leet_dict = {
        "a": "4",
        "b": "8",
        "c": "<",
        "d": "|)",
        "e": "3",
        "f": "|=",
        "g": "9",
        "h": "#",
        "i": "1",
        "j": "_|",
        "k": "|<",
        "l": "|_",
        "m": "/\\/\\",
        "n": "^/",
        "o": "0",
        "p": "|*",
        "q": "(,)",
        "r": "|2",
        "s": "5",
        "t": "+",
        "u": "|_|",
        "v": "\\/",
        "w": "\\^/",
        "x": "%",
        "y": "`/",
        "z": "2",
    }
    return "".join(leet_dict.get(char, char) for char in text.lower())


def binary_encode(text):
    """
    Encode the text using binary.
    """
    return " ".join(format(ord(char), "08b") for char in text)


def hex_encode(text):
    """
    Encode the text using hexadecimal.
    """
    return " ".join(format(ord(char), "02x") for char in text)


def base64_encode(text):
    """
    Encode the text using Base64.
    """
    import base64

    return base64.b64encode(text.encode()).decode()


def rot13_encode(text):
    """
    Encode the text using ROT13.
    """
    import codecs

    return codecs.encode(text, "rot_13")


def reverse_encode(text):
    """
    Encode the text by reversing it.
    """
    return text[::-1]



def atbash_decode(encoded_text):
    """
    Decode the text from the Atbash cipher.
    Atbash is a reciprocal cipher, so decoding is the same as encoding.
    """
    return atbash_encode(encoded_text)

def caesar_decode(encoded_text, shift):
    """
    Decode the text from the Caesar cipher by shifting in the opposite direction.
    """
    # Decoding is the same as encoding with a negative shift.
    return caesar_encode(encoded_text, -shift)

def vigenere_decode(encoded_text, key):
    """
    Decode the text from the Vigenère cipher.
    """
    decrypted_text = []
    key = key.lower()
    key_length = len(key)
    text = encoded_text.lower()
    
    for i, char in enumerate(text):
        if char.isalpha():
            # The decryption formula is C = (E - K + 26) % 26
            encoded_char_val = ord(char) - ord('a')
            key_char_val = ord(key[i % key_length]) - ord('a')
            decrypted_char_val = (encoded_char_val - key_char_val + 26) % 26
            decrypted_char = chr(decrypted_char_val + ord('a'))
            decrypted_text.append(decrypted_char)
        else:
            decrypted_text.append(char)
            
    return "".join(decrypted_text)


def braille_decode(encoded_text):
    """
    Decode the text from Braille.
    """
    braille_dict = {
        "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑", "f": "⠋", "g": "⠛",
        "h": "⠓", "i": "⠊", "j": "⠚", "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝",
        "o": "⠕", "p": "⠏", "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞", "u": "⠥",
        "v": "⠧", "w": "⠺", "x": "⠭", "y": "⠽", "z": "⠵",
    }
    # Create a reversed dictionary for decoding
    reversed_braille_dict = {v: k for k, v in braille_dict.items()}
    return "".join(reversed_braille_dict.get(char, char) for char in encoded_text)


def morse_decode(encoded_text):
    """
    Decode the text from Morse code. Assumes letters are space-separated.
    """
    morse_dict = {
        "a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.",
        "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..",
        "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
        "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-",
        "y": "-.--", "z": "--..",
    }
    # Create a reversed dictionary for decoding
    reversed_morse_dict = {v: k for k, v in morse_dict.items()}
    words = encoded_text.strip().split(' ')
    return "".join(reversed_morse_dict.get(word, ' ') for word in words)

def pig_latin_decode(encoded_text):
    """
    Decode the text from Pig Latin.
    """
    def pig_latin_decode_word(word):
        if word.endswith("yay"):
            return word[:-3]
        elif word.endswith("ay"):
            # Move the last letter (before 'ay') to the front
            return word[-3] + word[:-3]
        else:
            return word # Not a valid Pig Latin word

    words = encoded_text.split()
    original_words = [pig_latin_decode_word(word) for word in words]
    return " ".join(original_words)

def leet_decode(encoded_text):
    """
    Decode the text from Leet Speak.
    """
    leet_dict = {
        "a": "4", "b": "8", "c": "<", "d": "|)", "e": "3", "f": "|=", "g": "9",
        "h": "#", "i": "1", "j": "_|", "k": "|<", "l": "|_", "m": "/\\/\\", "n": "^/",
        "o": "0", "p": "|*", "q": "(,)", "r": "|2", "s": "5", "t": "+", "u": "|_|",
        "v": "\\/", "w": "\\^/", "x": "%", "y": "`/", "z": "2",
    }
    # Create a reversed dictionary for decoding
    reversed_leet_dict = {v: k for k, v in leet_dict.items()}
    # Iteratively replace symbols. This is naive but works for this specific dict.
    decoded_text = encoded_text
    for symbol, letter in reversed_leet_dict.items():
        decoded_text = decoded_text.replace(symbol, letter)
    return decoded_text

def binary_decode(encoded_text):
    """
    Decode the text from binary.
    """
    binary_values = encoded_text.split(' ')
    ascii_string = "".join([chr(int(binary_val, 2)) for binary_val in binary_values])
    return ascii_string

def hex_decode(encoded_text):
    """
    Decode the text from hexadecimal.
    """
    hex_values = encoded_text.split(' ')
    ascii_string = "".join([chr(int(hex_val, 16)) for hex_val in hex_values])
    return ascii_string

def base64_decode(encoded_text):
    """
    Decode the text from Base64.
    """
    return base64.b64decode(encoded_text).decode()

def rot13_decode(encoded_text):
    """
    Decode the text from ROT13.
    ROT13 is its own inverse.
    """
    return rot13_encode(encoded_text)

def reverse_decode(encoded_text):
    """
    Decode the text by reversing it again.
    """
    return encoded_text[::-1]


### The Master Decoder ###

def prompt_decoder(encoded_text, encoding):
    """
    Decode the prompt from the specified encoding.
    """
    if encoding == "atbash":
        return atbash_decode(encoded_text)
    elif encoding == "caesar":
        return caesar_decode(encoded_text, 3)  # Use the same shift value
    elif encoding == "vigenere":
        return vigenere_decode(encoded_text, "KEY")  # Use the same key
    elif encoding == "braille":
        return braille_decode(encoded_text)
    elif encoding == "morse":
        return morse_decode(encoded_text)
    elif encoding == "pig_latin":
        return pig_latin_decode(encoded_text)
    elif encoding == "leet":
        return leet_decode(encoded_text)
    elif encoding == "binary":
        return binary_decode(encoded_text)
    elif encoding == "hex":
        return hex_decode(encoded_text)
    elif encoding == "base64":
        return base64_decode(encoded_text)
    elif encoding == "rot13":
        return rot13_decode(encoded_text)
    elif encoding == "reverse":
        return reverse_decode(encoded_text)
    else:
        raise ValueError(f"Unsupported decoding type: {encoding}")


### Full Encoder Implementations (with Morse fix) ###
# (Pasting your original code here for a complete, runnable script)

def morse_encode(text):
    """
    Encode the text using Morse code.
    MODIFIED: Added a space separator to make it decodable.
    """
    morse_dict = {
        "a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.",
        "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..",
        "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
        "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-",
        "y": "-.--", "z": "--..",
    }
    # Join with spaces to allow for unambiguous decoding
    return " ".join(morse_dict.get(char, char) for char in text.lower() if char in morse_dict)

# (The rest of your encoder functions go here...)
def atbash_encode(text):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    reversed_alphabet = alphabet[::-1]
    return text.lower().translate(str.maketrans(alphabet, reversed_alphabet))

def caesar_encode(text, shift):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    return text.lower().translate(str.maketrans(alphabet, shifted_alphabet))

def vigenere_encode(text, key):
    encrypted_text = []
    key = key.lower()
    key_length = len(key)
    text = text.lower()
    for i, char in enumerate(text):
        if char.isalpha():
            text_char_val = ord(char) - ord('a')
            key_char_val = ord(key[i % key_length]) - ord('a')
            encrypted_char_val = (text_char_val + key_char_val) % 26
            encrypted_text.append(chr(encrypted_char_val + ord('a')))
        else:
            encrypted_text.append(char)
    return "".join(encrypted_text)
    
def braille_encode(text):
    braille_dict = {"a": "⠁","b": "⠃","c": "⠉","d": "⠙","e": "⠑","f": "⠋","g": "⠛","h": "⠓","i": "⠊","j": "⠚","k": "⠅","l": "⠇","m": "⠍","n": "⠝","o": "⠕","p": "⠏","q": "⠟","r": "⠗","s": "⠎","t": "⠞","u": "⠥","v": "⠧","w": "⠺","x": "⠭","y": "⠽","z": "⠵"}
    return "".join(braille_dict.get(char, char) for char in text.lower())

def pig_latin_encode(text):
    def pig_latin_word(word):
        if not word or not word.isalpha(): return word
        if word[0] in "aeiou": return word + "yay"
        else: return word[1:] + word[0] + "ay"
    words = text.lower().split()
    pig_latin_words = [pig_latin_word(word) for word in words]
    return " ".join(pig_latin_words)

def leet_encode(text):
    leet_dict = {"a": "4","b": "8","c": "<","d": "|)","e": "3","f": "|=","g": "9","h": "#","i": "1","j": "_|","k": "|<","l": "|_","m": "/\\/\\","n": "^/","o": "0","p": "|*","q": "(,)","r": "|2","s": "5","t": "+","u": "|_|","v": "\\/","w": "\\^/","x": "%","y": "`/", "z": "2"}
    return "".join(leet_dict.get(char, char) for char in text.lower())

def binary_encode(text):
    return " ".join(format(ord(char), "08b") for char in text)

def hex_encode(text):
    return " ".join(format(ord(char), "02x") for char in text)

def base64_encode(text):
    return base64.b64encode(text.encode()).decode()

def rot13_encode(text):
    return codecs.encode(text, "rot_13")

def reverse_encode(text):
    return text[::-1]
    
# Master Encoder from your code
def prompt_encoder(prompt, encoding):
    if encoding == "atbash": return atbash_encode(prompt)
    elif encoding == "caesar": return caesar_encode(prompt, 3)
    elif encoding == "vigenere": return vigenere_encode(prompt, "KEY")
    elif encoding == "braille": return braille_encode(prompt)
    elif encoding == "morse": return morse_encode(prompt)
    elif encoding == "pig_latin": return pig_latin_encode(prompt)
    elif encoding == "leet": return leet_encode(prompt)
    elif encoding == "binary": return binary_encode(prompt)
    elif encoding == "hex": return hex_encode(prompt)
    elif encoding == "base64": return base64_encode(prompt)
    elif encoding == "rot13": return rot13_encode(prompt)
    elif encoding == "reverse": return reverse_encode(prompt)
    else: raise ValueError("Unsupported encoding type.")

