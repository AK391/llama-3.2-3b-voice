import gradio as gr
import numpy as np
import io
from pydub import AudioSegment
import tempfile
import os
import base64
import openai
from dataclasses import dataclass, field
from threading import Lock

# Lepton API setup
client = openai.OpenAI(
    base_url="https://llama3-1-8b.lepton.run/api/v1/",
    api_key=os.environ.get('LEPTON_API_TOKEN')
)

@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

def transcribe_audio(audio):
    # This is a placeholder function. In a real-world scenario, you'd use a
    # speech-to-text service here. For now, we'll just return a dummy transcript.
    return "This is a dummy transcript. Please implement actual speech-to-text functionality."

def generate_response_and_audio(message, state):
    with state.lock:
        state.conversation.append({"role": "user", "content": message})
        
        completion = client.chat.completions.create(
            model="llama3-1-8b",
            messages=state.conversation,
            max_tokens=128,
            stream=True,
            extra_body={
                "require_audio": "true",
                "tts_preset_id": "jessica",
            }
        )

        full_response = ""
        audio_chunks = []

        for chunk in completion:
            if not chunk.choices:
                continue
            
            content = chunk.choices[0].delta.content
            audio = getattr(chunk.choices[0], 'audio', [])
            
            if content:
                full_response += content
                yield full_response, None, state
            
            if audio:
                audio_chunks.extend(audio)
                audio_data = b''.join([base64.b64decode(a) for a in audio_chunks])
                yield full_response, audio_data, state

        state.conversation.append({"role": "assistant", "content": full_response})

def chat(message, state):
    if not message:
        return "", None, state

    return generate_response_and_audio(message, state)

def process_audio(audio, state):
    if audio is None:
        return "", state
    
    # Convert numpy array to wav
    audio_segment = AudioSegment(
        audio[1].tobytes(),
        frame_rate=audio[0],
        sample_width=audio[1].dtype.itemsize,
        channels=1 if len(audio[1].shape) == 1 else audio[1].shape[1]
    )
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_segment.export(temp_audio.name, format="wav")
        transcript = transcribe_audio(temp_audio.name)
    
    os.unlink(temp_audio.name)
    
    return transcript, state

with gr.Blocks() as demo:
    state = gr.State(AppState())
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(source="microphone", type="numpy")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            text_input = gr.Textbox(show_label=False, placeholder="Type your message here...")
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio")
    
    audio_input.change(process_audio, [audio_input, state], [text_input, state])
    text_input.submit(chat, [text_input, state], [chatbot, audio_output, state])
    
demo.launch()