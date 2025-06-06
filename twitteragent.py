import gradio as gr
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini
import google.generativeai as genai
import tweepy
import config

# Initialize the agent
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGo(fixed_max_results=3)],
    show_tool_calls=True
)

# Set up Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)

# Set up Tweepy client
tweepy_client = tweepy.Client(
    consumer_key=config.CONSUMER_KEY,
    consumer_secret=config.CONSUMER_SECRET,
    access_token=config.ACCESS_TOKEN,
    access_token_secret=config.ACCESS_SECRET
)

# Define the Gradio function
def generate_and_tweet(input_query):
    context = agent.run(input_query, markdown=True)

    response = genai.generate_content(
        model="gemini-2.0-flash",
        contents=f"Generate a tweet based on this: {context}",
    )
    tweet = response.text.strip()

    tweet_response = tweepy_client.create_tweet(text=tweet)

    return f"✅ Tweet posted successfully:\n\n{tweet}"

# Gradio interface
gr.Interface(
    fn=generate_and_tweet,
    inputs=gr.Textbox(label="Enter a topic for the tweet"),
    outputs=gr.Textbox(label="Tweet Status"),
    title="AI Twitter Bot",
    description="Enter a topic. The AI will research, generate a tweet, and post it to Twitter."
).launch()
