from fastapi import FastAPI
import gradio as gr

from app import demo

fastapi_app = FastAPI()

app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/",
    root_path="/api"
)