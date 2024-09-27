import gradio as gr
import numpy as np
import io
from pydub import AudioSegment
import tempfile
import os
import base64
import openai
import time
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_start: float | None = None
    last_speech: float = 0
    conversation: list = field(default_factory=list)
    client: openai.OpenAI = None
    output_format: str = "mp3"

# Global lock for thread safety
state_lock = Lock()

def create_client(api_key):
    return openai.OpenAI(
        base_url="https://llama3-1-8b.lepton.run/api/v1/",
        api_key=api_key
    )

def process_audio(audio: tuple, state: AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
        state.last_speech = time.time()
    else:
        state.stream = np.concatenate((state.stream, audio[1]))

    # Improved pause detection
    current_time = time.time()
    if np.max(np.abs(audio[1])) > 0.1:  # Adjust this threshold as needed
        state.last_speech = current_time
        state.pause_start = None
    elif state.pause_start is None:
        state.pause_start = current_time

    # Check if pause is long enough to stop recording
    if state.pause_start and (current_time - state.pause_start > 2.0):  # 2 seconds of silence
        return gr.Audio(recording=False), state

    return None, state

def generate_response_and_audio(audio_bytes: bytes, state: AppState):
    if state.client is None:
        raise gr.Error("Please enter a valid API key first.")

    format_ = state.output_format
    bitrate = 128 if format_ == "mp3" else 32  # Higher bitrate for MP3, lower for OPUS
    audio_data = base64.b64encode(audio_bytes).decode()
    
    try:
        stream = state.client.chat.completions.create(
            extra_body={
                "require_audio": True,
                "tts_preset_id": "jessica",
                "tts_audio_format": format_,
                "tts_audio_bitrate": bitrate
            },
            model="llama3.1-8b",
            messages=[{"role": "user", "content": [{"type": "audio", "data": audio_data}]}],
            temperature=0.7,
            max_tokens=256,
            stream=True,
        )

        full_response = ""
        audios = []

        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            audio = getattr(chunk.choices[0], 'audio', [])
            if content:
                full_response += content
                yield full_response, None, state
            if audio:
                audios.extend(audio)

        final_audio = b''.join([base64.b64decode(a) for a in audios])

        state.conversation.append({"role": "user", "content": "Audio input"})
        state.conversation.append({"role": "assistant", "content": full_response})

        yield full_response, final_audio, state

    except Exception as e:
        raise gr.Error(f"Error during audio streaming: {e}")

def response(state: AppState):
    if state.stream is None or len(state.stream) == 0:
        return None, None, state
    
    audio_buffer = io.BytesIO()
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")

    generator = generate_response_and_audio(audio_buffer.getvalue(), state)
    
    # Process the generator to get the final results
    final_text = ""
    final_audio = None
    for text, audio, updated_state in generator:
        final_text = text if text else final_text
        final_audio = audio if audio else final_audio
        state = updated_state

    # Update the chatbot with the final conversation
    chatbot_output = state.conversation[-2:]  # Get the last two messages (user input and AI response)
    
    # Reset the audio stream for the next interaction
    state.stream = None
    state.pause_start = None
    state.last_speech = 0
    
    return chatbot_output, final_audio, state

def set_api_key(api_key, state):
    if not api_key:
        raise gr.Error("Please enter a valid API key.")
    state.client = create_client(api_key)
    return "API key set successfully!", state

def update_format(format, state):
    state.output_format = format
    return state

with gr.Blocks() as demo:
    with gr.Row():
        api_key_input = gr.Textbox(type="password", label="Enter your Lepton API Key")
        set_key_button = gr.Button("Set API Key")
    
    api_key_status = gr.Textbox(label="API Key Status", interactive=False)
    
    with gr.Row():
        format_dropdown = gr.Dropdown(choices=["mp3", "opus"], value="mp3", label="Output Audio Format")
    
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Input Audio", sources="microphone", type="numpy")
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            output_audio = gr.Audio(label="Output Audio", autoplay=True)
    
    state = gr.State(AppState())

    set_key_button.click(set_api_key, inputs=[api_key_input, state], outputs=[api_key_status, state])
    format_dropdown.change(update_format, inputs=[format_dropdown, state], outputs=[state])

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.25,  # Reduced to make it more responsive
        time_limit=60,  # Increased to allow for longer messages
    )
    
    respond = input_audio.stop_recording(
        response,
        [state],
        [chatbot, output_audio, state]
    )

demo.launch()