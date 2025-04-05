import streamlit as st
from utils import load_chroma_db, load_llm, create_qa_chain, create_prompt_template


chroma_db = load_chroma_db()
llm = load_llm()
PROMPT = create_prompt_template()
qa = create_qa_chain(chroma_db, llm, PROMPT)


st.title("Acıbadem Üniversitesi Asistanı")
st.write("https://www.youtube.com/watch?v=N2Xx_P0a9lo")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


query = st.text_input("Sorunuzu yazın:")

answer = None

if st.button("Yanıt Al"):
    if query:
        try:

            retriever = chroma_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            relevant_docs = retriever.get_relevant_documents(query)


            context = " ".join([doc.page_content for doc in relevant_docs])


            response = qa.invoke({"context": context, "query": query})
            answer = response.get("result", "Yanıt bulunamadı.")


            st.session_state.chat_history.insert(0, {"question": query, "answer": answer})

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

    else:
        st.warning("Lütfen bir soru yazın!")

#
if answer:
    st.subheader("Yanıt:")
    st.write(answer)


st.subheader("Sohbet Geçmişi")
for chat in st.session_state.chat_history:
    st.write(f"**Soru:** {chat['question']}")
    st.write(f"**Yanıt:** {chat['answer']}")
    st.write("---")
