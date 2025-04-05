#import ollama
from ollama import chat
from ollama import ChatResponse
from warnings import filterwarnings

filterwarnings("ignore")


def chat_reponse(input_message):
    
    user_message_info = f"""
    \nOgrencinin sordugu soru: {input_message} \n
    """
    user_message_info += 'benim bu soru ile ilgili buldugum chunk listesi {mesaj}'
    
    response: ChatResponse = chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': f"{user_message_info}",
            },
        ])
        
    content = response['message']['content']
    
    return content
    
print(chat_reponse('Okul kartimi kaybettim ne yapabilirim?'))