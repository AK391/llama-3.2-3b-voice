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
    pause_detected: bool = False
    started_talking: bool = False
    stopped: bool = False
    conversation: list = field(default_factory=list)
    client: openai.OpenAI = None

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
    else:
        state.stream = np.concatenate((state.stream, audio[1]))

    # Simple pause detection (you might want to implement a more sophisticated method)
    if len(state.stream) > state.sampling_rate * 0.5:  # 0.5 second of silence
        state.pause_detected = True
        return gr.Audio(recording=False), state
    return None, state

def generate_response_and_audio(audio_bytes: bytes, state: AppState):
    if state.client is None:
        raise gr.Error("Please enter a valid API key first.")

    format_ = "opus"
    bitrate = 16
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
            temperature=0.5,
            max_tokens=128,
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
                audio_data = b''.join([base64.b64decode(a) for a in audios])
                yield full_response, audio_data, state

        state.conversation.append({"role": "user", "content": "Audio input"})
        state.conversation.append({"role": "assistant", "content": full_response})

    except Exception as e:
        raise gr.Error(f"Error during audio streaming: {e}")

def response(state: AppState):
    if not state.pause_detected:
        return None, None, AppState()
    
    audio_buffer = io.BytesIO()
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")

    return generate_response_and_audio(audio_buffer.getvalue(), state)

def set_api_key(api_key, state):
    if not api_key:
        raise gr.Error("Please enter a valid API key.")
    state.client = create_client(api_key)
    return "API key set successfully!", state

def start_recording_user(state: AppState):
    if not state.stopped:
        return gr.Audio(recording=True)

with gr.Blocks() as demo:
    with gr.Row():
        api_key_input = gr.Textbox(type="password", label="Enter your Lepton API Key")
        set_key_button = gr.Button("Set API Key")
    
    api_key_status = gr.Textbox(label="API Key Status", interactive=False)
    
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Input Audio", sources="microphone", type="numpy")
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
    
    state = gr.State(AppState())

    set_key_button.click(set_api_key, inputs=[api_key_input, state], outputs=[api_key_status, state])

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.50,
        time_limit=30,
    )
    
    respond = input_audio.stop_recording(
        response,
        [state],
        [chatbot, output_audio, state]
    )

    restart = output_audio.stop(
        start_recording_user,
        [state],
        [input_audio]
    )
    
    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(stopped=True), gr.Audio(recording=False)),
        None,
        [state, input_audio],
        cancels=[respond, restart]
    )

demo.launch()