import gradio as gr

def dummy_chat(message, history=None):
    return "Hello! This is a demo response.", ""

with gr.Blocks(title="Multimodal AI Chat") as demo:
    gr.Markdown("# 🛡️ Aegis Multimodal AI Demo")
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="Your Message")
    submit = gr.Button("Send")
    msg.submit(dummy_chat, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
