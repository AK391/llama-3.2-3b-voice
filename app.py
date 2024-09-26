import base64
import gradio as gr
import openai
from pydub import AudioSegment
import io
import tempfile
import speech_recognition as sr

def transcribe_audio(audio):
    # Convert the audio to wav format
    audio = AudioSegment.from_file(audio)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Save as wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio.export(temp_audio.name, format="wav")
        temp_audio_path = temp_audio.name

    # Perform speech recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    # Clean up the temporary file
    os.unlink(temp_audio_path)

    return text

def process_audio(audio, api_token):
    if not api_token:
        return "Please provide an API token.", None

    # Initialize the OpenAI client with the user-provided token
    client = openai.OpenAI(
        base_url="https://llama3-2-3b.lepton.run/api/v1/",
        api_key=api_token
    )

    # Transcribe the input audio
    transcription = transcribe_audio(audio)

    try:
        # Process the transcription with the API
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": transcription},
            ],
            max_tokens=128,
            stream=True,
            extra_body={
                "require_audio": "true",
                "tts_preset_id": "jessica",
            }
        )

        response_text = ""
        audios = []

        for chunk in completion:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            audio = getattr(chunk.choices[0], 'audio', [])
            if content:
                response_text += content
            if audio:
                audios.extend(audio)

        # Combine audio chunks and save as MP3
        audio_data = b''.join([base64.b64decode(audio) for audio in audios])
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        return response_text, temp_audio_path

    except Exception as e:
        return f"An error occurred: {str(e)}", None

# Create the Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Textbox(label="API Token", type="password")
    ],
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio")
    ],
    title="Audio-to-Audio Demo",
    description="Upload an audio file and provide your API token to get a response in both text and audio format."
)

# Launch the interface
iface.launch()