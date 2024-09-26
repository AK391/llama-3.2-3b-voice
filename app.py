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

@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    client: openai.OpenAI = None

def create_client(api_key):
    return openai.OpenAI(
        base_url="https://llama3-1-8b.lepton.run/api/v1/",
        api_key=api_key
    )

def transcribe_audio(audio):
    # This is a placeholder function. In a real-world scenario, you'd use a
    # speech-to-text service here. For now, we'll just return a dummy transcript.
    return "This is a dummy transcript. Please implement actual speech-to-text functionality."

def generate_response_and_audio(message, state):
    if state.client is None:
        raise gr.Error("Please enter a valid API key first.")

    with state.lock:
        state.conversation.append({"role": "user", "content": message})
        
        try:
            completion = state.client.chat.completions.create(
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
        except Exception as e:
            raise gr.Error(f"Error generating response: {str(e)}")

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

def set_api_key(api_key, state):
    if not api_key:
        raise gr.Error("Please enter a valid API key.")
    state.client = create_client(api_key)
    return "API key set successfully!", state

with gr.Blocks() as demo:
    state = gr.State(AppState())
    
    with gr.Row():
        api_key_input = gr.Textbox(type="password", label="Enter your Lepton API Key")
        set_key_button = gr.Button("Set API Key")
    
    api_key_status = gr.Textbox(label="API Key Status", interactive=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(source="microphone", type="numpy")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            text_input = gr.Textbox(show_label=False, placeholder="Type your message here...")
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio")
    
    set_key_button.click(set_api_key, inputs=[api_key_input, state], outputs=[api_key_status, state])
    audio_input.change(process_audio, inputs=[audio_input, state], outputs=[text_input, state])
    text_input.submit(chat, inputs=[text_input, state], outputs=[chatbot, audio_output, state])
    
demo.launch()