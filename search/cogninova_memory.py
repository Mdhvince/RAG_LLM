from collections import deque


class CogninovaMemory:
    def __init__(self):
        self.length = 3  # length is fixe to 3. follow-up question appear after one interaction.
        self.memory = deque(maxlen=self.length)


    def update(self, question_or_answer):
        """
        Update the memory with the question or answer
        """
        self.memory.append(question_or_answer)


    def clear(self):
        """
        Clear the memory
        Useful when the user want to start a new conversation. (Keep the same memory object)
        """
        self.memory.clear()


    def optimize(self, standalone_question):
        """
        Optimize the memory by removing the oldest question and answer and replace them with the standalone question
        :param standalone_question: The standalone question generated by the llm
        """
        self.clear()
        self.update(standalone_question)


    def is_full(self):
        """
        Check if the memory is full
        """
        return len(self.memory) == self.memory.maxlen


    def get_chat_history(self) -> str:
        """
        This will take the history of the conversation (self.memory) and the question asked by the user, combine them
        and generate a string that allow the llm later to generate a standalone question.

        Example input to llm for generating standalone question:

        Given the following conversation (chat history) and follow-up question, rephrase the follow-up question to be a
        standalone question.
        Chat history:
        Human: What is the capital of France?
        Assistant: Paris.
        Human: How far is it from Germany in kilometers?
        Assistant: 1000 km.
        Follow-up question: Can I get there by train?
        Your Standalone question:
        ----------------------------------------

        In the above example, a good standalone question would be: "Can I get from Paris to Germany by train?"

        Here we return only the chat history + follow-up question part
        """
        assert len(self.memory) == self.length, "Memory must have at least 2 elements (question and answer)"

        # Here we assume that the last element in the memory is the human's question, because the memory length is odd
        # and this function will be call only if the memory is full. I also assume the follow-up question to be the
        # last question in the memory

        chat_history = ""
        for i in range(0, len(self.memory), 2):
            try:
                chat_history += f"Human: {self.memory[i]}\nAssistant: {self.memory[i+1]}\n"
            except IndexError:
                chat_history += f"Follow-up question: {self.memory[-1]}"

        return chat_history



if __name__ == "__main__":
    memory = deque(maxlen=3)
    memory.extend(["human", "ai assistant", "human fq"])
    chat_history = ""
    for i in range(0, len(memory), 2):
        try:
            chat_history += f"Human: {memory[i]}\nAssistant: {memory[i + 1]}\n"
        except IndexError:
            chat_history += f"Follow-up question: {memory[-1]}\n"

    print(chat_history)