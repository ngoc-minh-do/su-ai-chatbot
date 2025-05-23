import gradio as gr


def generate_response(prompt_txt):
    return prompt_txt


chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Su AI Chatbot",
    description="Ask any question and the chatbot will try to answer.",
)

# Launch the app
chat_application.launch(server_name="0.0.0.0", server_port=7860)
