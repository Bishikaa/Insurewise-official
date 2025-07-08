# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# import os

# load_dotenv()
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key = GOOGLE_API_KEY)


# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")



# def vectorstore(collection_name, directory):
#     vector_store = Chroma(
#         collection_name=collection_name,
#         persist_directory= directory,
#         embedding_function= embeddings
#     )
#     return vector_store

# vs1 = vectorstore('insurance1.vdb','insurance1.db')
# vs2 = vectorstore('insurance2.vdb','insurance2.db')
# vs3 = vectorstore('insurance3.vdb','insurance3.db')
# vs4 = vectorstore('insurance4.vdb','insurance4.db')
# vs5 = vectorstore('insurance5.vdb','insurance5.db')

# def retrieve(question):
#     retriever1 = vs1.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     retriever2 = vs2.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     retriever3 = vs3.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     retriever4 = vs4.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     retriever5 = vs5.as_retriever(search_type='similarity', search_kwargs={'k': 5})

#     result1 = retriever1.invoke(question)
#     result2 = retriever2.invoke(question)
#     result3 = retriever3.invoke(question)
#     result4 = retriever4.invoke(question)
#     result5 = retriever5.invoke(question)

#     # print(result1) 
#     # -------- Break --------

#     # print(result2)

#     # ---- Break -----

#     # print(result3) 
#     # -------- Break --------

#     # print(result4)

#     # ---- Break -----

#     # print(result5) 
#     # -------- Break --------


#     # ---- Break -----
#     return result1, result2, result3, result4, result5


# def generateresponse(context1, context2, context3, context4, context5, question ,chat_history):
#     prompt = PromptTemplate.from_template(
#         '''
#     You are a friendly AI-powered **Insurance Recommendation System**.
#     Your task is to provide the most suitable insurance recommendation based on the user's question. 
#     You will receive **detailed insurance policy information (contexts)** from five different companies. Each context will describe the company's plans, coverage, pricing, benefits, eligibility, terms, and conditions.

#     ---
#     ### ADDITIONAL CONTEXT:
#     You also have access to the user's previous conversation (chat history). **Use it to understand their preferences, confusions, or previously asked questions**. This will help you give a more personalized and accurate answer.

#     Here’s the past chat history (if available):

#     ```
#     {chat_history}
#     ```

#     If there's no useful info in the history, simply answer based on the current question and context.

#     ---

#     ### YOUR OBJECTIVE:

#     1. **Act as a Human Insurance Advisor:**
#     Act like a professional assistant trained by all five companies. Be supportive, knowledgeable, and easy to understand and friendly — especially for someone who is completely new to insurance.
    
#     2. **Recommend First, Compare Only If Needed:**
#     Your job is to **recommend the best-matching insurance policy** based on the user's question in detail.  
#     - If the question **asks for a comparison**, then compare the five contexts clearly.
#     - But if the question only asks **which is the best**, **go straight to the most suitable option** and also explain why it’s the best fit in detail.
#     - Only mention other companies briefly or if they have something significantly better or worth avoiding.

#     3. **Use Contexts ONLY:**
#     Only use information from the five provided contexts. **Do not guess, assume, or invent any new details**. If the answer is missing, use only basic and verified insurance knowledge — and **say so clearly**.

#     ---

#     ### HOW TO THINK & RESPOND:

#     4. **Understand, then Explain:**
#     First read and understand each context fully. Then, **rephrase and explain** what is relevant in **your own words**, using **clear and beginner-friendly language**.
#       - **Never copy sentences** directly from the context.
#       - Explain as if the user has no background in insurance.
#       - Avoid technical jargon unless necessary. If used, define it briefly.

#     5. **Be Concise but Complete:**
#     Give full answers without being repetitive. Focus only on what’s important. No filler, no unnecessary words.

#     6. **Stay Organized:**
#     Use short paragraphs or bullet points if needed. Break down your response clearly so it’s easy to read.

#     ---

#     ### FOCUS ON BEING USEFUL

#     7. **Answer Follow-Up Questions Too:**
#     Be ready to answer follow-up questions related to insurance (e.g., what is a premium, what is term life insurance). Always keep the explanation simple and accurate.

#     8. **Professional & Friendly Tone:**
#     Sound like a polite and well-trained representative. Be respectful, informative, and warm. Don’t sound robotic.

#     9. **Avoid Hallucination:**
#     Do NOT make up prices, features, benefits, or names that are not in the context. If something is missing, say **"I do not have information on that specific plan. Based on the available information, I recommend [the best option]."**

#     10. **Do Not Recommend Contacting the Company Unless Needed:**
#     Only say "check the website" or "contact the company" if absolutely necessary. If you must, say it softly like:
#       *"For more updated or detailed info, you may also visit SuryaJyoti's official website: https://suryajyotilife.com"*

#     11. **Compare Fairly & Give Clear Reason:**
#     Even if multiple policies seem similar, try to **explain what makes one of them slightly better** based on the user’s question. If more than one are equally good, say that and explain why in detail.

#     12. **Be Honest About Limitations:**
#     If no exact match can be found, suggest the closest option and explain why it's the next best in detail. Always be transparent.

#     13. **Anticipate User Confusion:**
#     If something in the user query seems vague or unclear, gently point it out and try to give the best possible answer with what is given.
     
#     14. Always explai`n **why** the recommended policy is the best fit. Do not just name the company — describe what specific benefits, coverage, or features make it the right match for the user's question.

#     16. Speak with a natural, helpful tone — like a human assistant trying to make sure the user fully understands their options before choosing and end your answer in a freindly way as well.

#     17. **Important Formatting Guidelines**:
#   - All responses must be structured using **proper Markdown formatting**.
#   - The final output should be clear, organized, and user-friendly.
#   - Use the following format where applicable:
#     - **Tables** for comparisons, datasets, parameters, pros/cons, etc.
#     - **Lists** (ordered/unordered) for steps, tips, items, summaries, or instructions.
#     - **Paragraphs** for descriptions, explanations, or contextual information.

#   **Visual & Textual Clarity Requirements**:
#   - Use **bold** and *italic* text appropriately to emphasize key points.
#   - Use headings (`#`, `##`, `###`) to organize sections and improve readability.
#   - Maintain **consistent and readable font** through proper Markdown styling.
#   - Keep responses **concise, relevant, and logically structured**.
#   - Include **code blocks** (using triple backticks) for any code-related output.

#   Your goal is to generate responses that are not only factually correct but also **well-presented, skimmable, and visually engaging**.


#        ---

#     ### COMPANY NAMES:
#     - Company 1: [SURYAJYOTI]
#     - Company 2: [LIC NEPAL]
#     - Company 3: [SANIMA]
#     - Company 4: [MET LIFE]
#     - Company 5: [SUNLIFE]

#     These are the 5 context documents:
#     {context1}
#     {context2}
#     {context3}
#     {context4}
#     {context5}

#     User's question:
#     {question}
#     '''
#     )

#     final_prompt = prompt.format(context1=context1, context2=context2, context3=context3, context4=context4, context5=context5, question=question,chat_history = chat_history)
#     response = llm.invoke(final_prompt)
#     return response.content


# def format_chat_history(history):
#     if not history:
#         return "None"
#     return '\n'.join([f"User_query: {item['user_query']}\nAI: {item['chatbot answer']}" for item in history])

# history = []
# formatted_history = format_chat_history(history)


# def get_chatbot_response(user_message):
#     output1, output2, output3, output4, output5 = retrieve(user_message)

#     context1 = ''.join(doc.page_content + '\n' for doc in output1)
#     context2 = ''.join(doc.page_content + '\n' for doc in output2)
#     context3 = ''.join(doc.page_content + '\n' for doc in output3)
#     context4 = ''.join(doc.page_content + '\n' for doc in output4)
#     context5 = ''.join(doc.page_content + '\n' for doc in output5)

#     response = generateresponse(context1, context2, context3, context4, context5, user_message, formatted_history)


#     history.append({
#         'user_query': user_message,
#         'ai': response
#     })

#     print(history)
#     return response

