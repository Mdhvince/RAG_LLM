


class CogninovaTemplate:
    def __init__(self):
        pass

    @property
    def stuff_template(self) -> str:
        """
        'context' and 'question' represents the input keys
        """
        template = "Use the following pieces of context to answer the human's question at the end. " \
                   "If you don't know the answer, just say that you don't know, don't try to make up an answer." \
                   "\nContext information is below:" \
                   "\n-------------------------------------" \
                   "\n{context}" \
                   "\n\nHuman's question: {question}" \
                   "\nHelpful Answer: "
        return template


    @property
    def refine_template_start(self) -> str:
        """
        'context' and 'question' represents the input keys
        """
        template = "\nContext information is below:" \
                   "\n-------------------------------------" \
                   "\n{context}" \
                   "\n-------------------------------------" \
                   "\nGiven the context information and not prior knowledge, answer any questions" \
                   "\nquestion: {question}" \
                   "\nHelpful Answer: "
        return template


    @property
    def refine_template_next(self) -> str:
        """
        'question', 'guess' and 'context' represents the input keys
        """
        template = "question: {question}" \
                   "\noriginal answer: {guess}" \
                   "\nWe have the opportunity to refine the original answer (only if needed) " \
                   "with the new context below." \
                   "\n-------------------------------------" \
                   "\n{context}" \
                   "\n-------------------------------------" \
                   "\nGiven the new context, refine the original answer to better answer the question. " \
                   "If the context is not useful, return the original answer." \
                   "\nRefined Answer: "
        return template


    @property
    def standalone_question_template(self) -> str:
        """
        'chat_history' represents the input key
        """
        template = "Given the following conversation (chat history) and follow-up question, rephrase the follow-up " \
                   "question to be a standalone question." \
                   "\nChat history:" \
                   "\n{chat_history}" \
                   "\n\nYour Standalone question: "
        return template

