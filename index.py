import json
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer

def create_chatbot():
    # Initialize the chatbot
    bot = ChatBot("Rudra", read_only=False, logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "default_response": "Sorry, I don't have an answer for this. Please use the FIX-DEVICE option to contact Machinevice.",
            "maximum_similarity_threshold": 0.9
        }
    ])
    return bot

def train_chatbot(bot):
    # Load training data from JSON file
    with open('training_data.json', 'r') as file:
        data = json.load(file)
        conversations = data['conversations']
    
    # Initialize the trainers
    corpus_trainer = ChatterBotCorpusTrainer(bot)
    list_trainer = ListTrainer(bot)

    # Train with ChatterBot corpus
    corpus_trainer.train("chatterbot.corpus.english")

    # Train the chatbot with custom data
    for conversation in conversations:
        list_trainer.train(conversation)

def chat_with_bot(bot):
    # Get a response
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Ending conversation.")
            break
        response = bot.get_response(user_input)
        print("Rudra: " + str(response))

# Main flow
if __name__ == "__main__":
    bot = create_chatbot()
    train_chatbot(bot)
    chat_with_bot(bot)
