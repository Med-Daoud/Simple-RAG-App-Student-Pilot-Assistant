
from retriever import get_retriever
from generator import generate_answer

def main():
    retriever = get_retriever()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        docs = retriever.invoke(question)
        answer = generate_answer(docs, question)

        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
