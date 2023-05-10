# JarvisBase
Question-answering chatbot using OpenAI's GPT-3.5-turbo model, DeepLake for the vector database, and the Whisper API for voice transcription. The chatbot also uses Eleven Labs to generate audio responses.

## Basic Idea
1. Get all the Huggingface Hub Python Library Articles 
2. Embed them with Deeplake 
3. Allow the user to record their voice or type their query
4. Generate the repsonse and make an audio recording using elevenlabs


## Installation
Clone the repository.

        git clone https://github.com/peterw/QnA.git

Install dependencies:

        pip install -r requirements.txt


Get your  [OpenAi API keys](https://platform.openai.com/account/api-keys), [Activeloop APi Keys](https://app.activeloop.ai/profile/kenyanroot/apitoken) and [Eleven Labs API Keys](https://beta.elevenlabs.io/speech-synthesis) and add them to your .env file.

## Usage
To set up and run this project, follow these steps:

1. Run the scrape.py script to embed the Intercom articles first

        python scrape.py

2. Start the app 

        streamlit run chat.py

Type your query in the input field and press enter.
If you have a microphone, you can click the record button and transcribe your audio. Click the transcribe button to get the text.
The bot will display the response in the chat history, and it will also be spoken using the Eleven Labs API.

## Sponsors
✨ Learn to build projects like this one (early bird discount): [BuildFast Course ](https://www.buildfastcourse.com/)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

