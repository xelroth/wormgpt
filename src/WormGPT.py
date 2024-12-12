from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from colorama import init, Fore, Style
import pandas as pd
import openai
import sqlite3


class InteractionManager:
    def __init__(self, db_name='interactions.db'):
        self.api_key = "key"
        self.db_name = db_name
        self.history = []
        self.last_question = None
        self.last_answer = None
        self.__CreateDatabase__()

    def __CreateDatabase__(self):
        """Creates the interactions database if it does not exist."""
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS interactions 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         question TEXT NOT NULL, 
                         answer TEXT NOT NULL, 
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()

    def __GetAllInteractions__(self):
        """Fetches all interactions from the database."""
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM interactions")
            return c.fetchall()

    def __AddInteraction__(self, question, answer):
        """Adds a new interaction to the database."""
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO interactions (question, answer) VALUES (?, ?)", (question, answer))
            conn.commit()

    def __ViewInteractions__(self):
        """Displays all interactions in a formatted manner."""
        interactions = self.__GetAllInteractions__()
        if not interactions:
            print(Fore.RED + "No interactions found." + Style.RESET_ALL)
            return
        print(Fore.MAGENTA + "\nQuestion-Answer Interactions:" + Style.RESET_ALL)
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
        for interaction in interactions:
            print(
            f"ID: {interaction[0]}",
            f"Question: {interaction[1]}",
            f"Answer: {interaction[2]}",
            f"Timestamp: {interaction[3]}"
            )
            print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

    def __GetApiKey__(self):
        """Retrieves the OpenAI API key."""
        return self.api_key

    def __GetAnswer__(self, api_key, question):
        """Fetches an answer from OpenAI API based on the provided question."""
        openai.api_key = api_key
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=question,
            temperature=0.7,
            max_tokens=4000,
            n=1,
            stop=["Human:", " AI:"]
        )
        return response['choices'][0]['text'].strip()

    def __AnalyzeInteractions__(self):
        """Analyzes interactions to discover patterns using LDA."""
        interactions = self.__GetAllInteractions__()
        if not interactions:
            print(Fore.RED + "No interactions found for analysis." + Style.RESET_ALL)
            return

        df = pd.DataFrame(interactions, columns=['id', 'question', 'answer', 'timestamp'])
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['question'] + " " + df['answer'])
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        print(Fore.MAGENTA + "\nDiscovered Patterns:" + Style.RESET_ALL)
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-10:]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Pattern {topic_idx + 1}: {', '.join(top_words)}")
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

    def __MainLoop__(self):
        """Main loop to handle user interactions."""
        init(autoreset=True)
        self.__DisplayIntro__()
        api_key = self.__GetApiKey__()

        while True:
            question = input(Fore.CYAN + "Enter your question: " + Style.RESET_ALL)
            if question.lower() == 'exit':
                print(Fore.YELLOW + "WormGPT welcomes you - See you soon." + Style.RESET_ALL)
                break
            elif question.lower() == 'view':
                self.__ViewInteractions__()
                continue

            answer = self.__GetAnswer__(api_key, question)
            print(Fore.GREEN + "\nAnswer:" + Style.RESET_ALL)
            print(Fore.MAGENTA + answer + Style.RESET_ALL)
            print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

            self.__AddInteraction__(question, answer)
            self.history.append({"question": question, "answer": answer})
            self.last_question = question
            self.last_answer = answer

    def __DisplayIntro__(self):
        """Displays the introductory message."""
        print(Fore.CYAN + """
 __      __                      _____________________________
/  \    /  \___________  _____  /  _____/\______   \__    ___/
\   \/\/   /  _ \_  __ \/     \/   \  ___ |     ___/ |    |
 \        (  <_> )  | \/  Y Y  \    \_\  \|    |     |    |
  \__/\  / \____/|__|  |__|_|  /\______  /|____|     |____|
       \/                    \/        \/

WormGPT V3.0 Ultimate Ⓡ

Welcome to the WormGPT. The biggest enemy of the well-known ChatGPT, lets talk to me!
        """ + Style.RESET_ALL)

if __name__ == '__main__':
    manager = InteractionManager()
    manager.__MainLoop__()
