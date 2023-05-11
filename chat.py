"""
This module demonstrates the usage of os, openai, and streamlit libraries.
"""

import os
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs import generate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

# Load environment variables from .env file and return the keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
eleven_api_key = os.environ.get("ELEVEN_API_KEY")
active_loop_data_set_path = os.environ.get("DEEPLAKE_DATASET_PATH")


def load_embeddings_and_database(active_loop_data_set_path):
    """
    Load embeddings and DeepLake database from the given path.

    Args:
        active_loop_data_set_path (str): The path to the active loop data set.

    Returns:
        tuple: A tuple containing the embeddings and the DeepLake database.
    """
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=active_loop_data_set_path,
        read_only=True,
        embedding_function=embeddings,
    )
    return db


def transcribe_audio(audio_file_path, openai_key):
    """
    Transcribe audio using OpenAI Whisper API.

    :param audio_file_path: The path to the audio file to be transcribed.
    :type audio_file_path: str
    :param openai_key: The API key for OpenAI.
    :type openai_key: str
    """
    openai.api_key = openai_key
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None


def record_and_transcribe_audio():
    """
    Record audio using audio_recorder and transcribe using transcribe_audio.

    Returns:
        transcription (str): The transcribed text from the recorded audio.
    """
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH, openai.api_key)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription


def display_transcription(transcription):
    """
    Display the transcription of the audio on the app.

    Args:
        transcription (str): The transcribed text from the recorded audio.
    """
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+", encoding='utf-8') as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")


def get_user_input(transcription):
    """
    Get user input from Streamlit text input field.

    Args:
        transcription (str): The transcribed text from the recorded audio.

    Returns:
        str: The user input text from the Streamlit text input field.
    """
    return st.text_input("", value=transcription if transcription else "", key="input")


def search_db(user_input, db):
    """
    Search the database for a response based on the user's query.

    Args:
        user_input (str): The user's input query.
        db (object): The database object to be searched.

    Returns:
        None
    """
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({"query": user_input})


def display_conversation(history):
    """
    Display conversation history using Streamlit messages.

    Args:
        history (dict): A dictionary containing the
        conversation history with keys "generated" and "past".

    Returns:
        None
    """
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))
        # Voice using Eleven API
        voice = "Bella"
        text = history["generated"][i]
        audio = generate(text=text, voice=voice, api_key=eleven_api_key)
        st.audio(audio, format="audio/mp3")


def main():
    """
    Main function to run the JarvisBase Streamlit app.

    Args:
        None

    Returns:
        None
    """
    # Initialize Streamlit app with a title
    st.write("# JarvisBase 🧙")

    # Load embeddings and the DeepLake database
    db = load_embeddings_and_database(active_loop_data_set_path)

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from text input or audio transcription
    user_input = get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        output = search_db(user_input, db)
        print(output["source_documents"])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
