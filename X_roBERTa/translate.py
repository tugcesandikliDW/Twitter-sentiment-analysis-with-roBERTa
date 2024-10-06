from googletrans import Translator

translator = Translator()



def translate_text(original_text: str) -> str:
    translation = translator.translate(original_text, dest='en')
    translated_text, original_language = translation.text, translation.src
    return translated_text, original_language

# original_text = "bu yemek çok tuzlu!"
# print(translate_text(original_text))