from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

class PromptCreator(ABC):
    # Factory/Creator: In this code, the factory is represented by the PromptCreator abstract base class (ABC) and its concrete subclasses (MainPromptCreator and IceCreamPromptCreator). 
    # The factory is responsible for creating instances of PromptTemplate objects based on certain conditions.
    @abstractmethod
    def create_prompt(self, context):pass


class IceCreamPromptCreator():
    #Concrete Creator: MainPromptCreator and IceCreamPromptCreator are concrete creators. 
    # They subclass PromptCreator and override the create_prompt() method to create instances of PromptTemplate with different templates and input variables, depending on the context.
    @staticmethod
    def create_prompt():
        ice_cream_assistant_template = """
        Question: {question} 
        Answer:
        """

        prompt = PromptTemplate(
            template=ice_cream_assistant_template,
            input_variables=["question"])
        
        return prompt  
    

class IceCreamPromptCreatorMemory():
    #Concrete Creator: MainPromptCreator and IceCreamPromptCreator are concrete creators. 
    # They subclass PromptCreator and override the create_prompt() method to create instances of PromptTemplate with different templates and input variables, depending on the context.
    @staticmethod
    def create_prompt():
        ice_cream_assistant_template = """
        Question: {question} 
        Answer:
        """

        prompt = PromptTemplate(
            template=ice_cream_assistant_template,
            input_variables=["chat_history", "question"])
        
        return prompt  
    

class IceCreamPromptCreatorAPI():
    #Concrete Creator: MainPromptCreator and IceCreamPromptCreator are concrete creators. 
    # They subclass PromptCreator and override the create_prompt() method to create instances of PromptTemplate with different templates and input variables, depending on the context.
    @staticmethod
    def create_prompt():
        api_url_template = """
                            Given the following API Documentation for Scoopsie's official ice cream store API: {api_docs}
                            Your task is to construct the most efficient API URL to answer 
                            the user's question, ensuring the call is optimized to include only necessary information.
                            Question: {question}
                            API URL:
        """
        
        api_url_prompt = PromptTemplate(input_variables=['api_docs', 'question'], template=api_url_template)

        api_response_template = """"
                            With the API Documentation for Scoopsie's official API: {api_docs} 
                            and the specific user question: {question} in mind,
                            and given this API URL: {api_url} for querying, here is the 
                            response from Scoopsie's API: {api_response}. 
                            Please provide a summary that directly addresses the user's question, 
                            omitting technical details like response format, and 
                            focusing on delivering the answer with clarity and conciseness, 
                            as if Scoopsie itself is providing this information.
                            Summary:
        """
        api_response_prompt = PromptTemplate(input_variables=['api_docs', 
                                                            'question', 
                                                            'api_url',
                                                            'api_response'],
                                     template=api_response_template)

        
        return api_url_prompt, api_response_prompt
    

