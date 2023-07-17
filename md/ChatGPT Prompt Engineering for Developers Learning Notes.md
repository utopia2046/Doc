# ChatGPT Prompt Engingeering for Developers Learning Notes

[Course link on DeepLearning.AI](https://learn.deeplearning.ai/chatgpt-prompt-eng/)

## Introduction

Target: As a developer, using API calls to LLM To quickly build software applications, rather than using current web interface (articles like 30 prompts everyone has to know)

Topics:

- Prompting best practices for software development
- Common use cases, summarizing, inferring, transforming, expanding
- Build a chatbot using an LLM

Two types of LLMs

- Base LLMs: Predict the next word based on text training data
- Instruction tuned LLMs: trained to follow instructions
  - start off with a base LLMs trained on a huge amount of text data
  - further refine using RLHF (reinforcement learning from human feedback)
  - goal: helpful, honest, harmless
  - important principles of prompting:
    - Write clear and specific instructions
    - Give the model time to think

## Guidelines

[Jupyter online notebook](https://s172-31-11-32p44983.lab-aws-production.deeplearning.ai/notebooks/l2-guidelines.ipynb)

``` console
pip install openai
```

``` python
import openai
openai.api_key = ''

# helper function
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
```

### Principle 1: Write clear and specific instructions

- Tactic 1: Use delimeters to avoid prompt injection attack
  - Triple quotes: """
  - Triple backticks: ```
  - Triple dashes: ---
  - Angle brackets: <>
  - XML tags: `<tag></tag>`
- Tactic 2: Ask for structured output (HTML, JSON, etc.)
- Tactic 3: Check whether conditions are satisfied, check assumptions required to do the task
- Tactic 4: Few-shot prompting: Give successful examples of completing tasks, then ask model to perform the task

``` python
# Tactic 1: use delimeters
text = f"""
"""
prompt = f"""
Summarize the text delimited by triple backticks \
into a single sentence.
```{text}```
"""
response = get_completion(prompt)

# Tactic 2: structured output
prompt = f"""
Generate a list of three made-up book titles along \
with their authors and genres.
Provide them in JSON format with the following keys:
book_id, title, author, genre.
"""
response = get_completion(prompt)

# Tactic 3: check condition
prompt = f"""
You will be provided with text delimited by triple quotes.
If it contains a sequence of instructions, \
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)

# Tactic 4: Few-shot prompting
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \
valley flows from a modest spring; the \
grandest symphony originates from a single note; \
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
```

### Principle 2: Give the model time to think

- Tactic 1: Specify the steps to complete a task
  - Step 1:
  - Step 2:
  - Step N:...
- Instruct the model to work out its own solution before rushing to a conclusion

``` python
# Tactic 1: Specify the steps required to complete a task
prompt_2 = f"""
Your task is to perform the following actions:
1 - Summarize the following text delimited by
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)

# Tactic 2: Instruct the model to work out its own solution
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem.
- Then compare your solution to the student's solution \
and evaluate if the student's solution is correct or not.
Don't decide if the student's solution is correct until
you have done the problem yourself.

Use the following format:
Question:
---
question here
---
Student's solution:
---
student's solution here
---
Actual solution:
---
steps to work out the solution and your solution here
---
Is the student's solution the same as actual solution \
just calculated:
---
yes or no
---
Student grade:
---
correct or incorrect
---

Question:
---
I'm building a solar power installation and I need help \
working out the financials.
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
---
Student's solution:
---
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
---
Actual solution:
"""
response = get_completion(prompt)
```

Model Limitations

- Hallucination: Makes statements that sound plausible but are not true

## Iterative

[Jupyter Notebook](https://s172-31-7-46p25072.lab-aws-production.deeplearning.ai/notebooks/l3-iterative-prompt-development.ipynb)

Iterative Prompt Development

- Idea
- Implementation(code/data), Prompt (clear & specific)
- Experimental result
- Error Analysis: analyze why result does not give desired output
- Refine the idea and the prompt
- Repeat

Iterative Process:

- Try somthing
- Analyze where the result does not give what you want
- Clarify instructions, give more time to think
- Refine prompts with a larger batch of examples

Example:

Issue 1: the text is too long.

prompt: `Use at most 50 words.`
prompt: `Use at most 3 sentences.`

| Tips: language model is not good at counting characters (because of tokenization), specifying characters number might not work.

Issue 2: text focuses on the wrong details

``` prompt
The description is intended for furniture retailers,
so should be technical in nature and focus on the
materials the product is constructed from.

At the end of the description, include every 7-character
Product ID in the technical specification.
```

Issue 3: Description needs a table of dimensions

``` prompt
After the description, include a table that gives the
product's dimensions. The table should have two columns.
In the first column include the name of the dimension.
In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website.
Place the description in a <div> element.
```

``` python
# view HTML in Jupyter
from IPython.display import display, HTML
display(HTML(response))
```

## Summarizing

``` prompt
Your task is to generate a short summary of a product review from an ecommerce site.
Summarize the review below, delimited by triple backticks, in at most 30 words.
```

Focus

``` prompt
Your task is to generate a short summary of a product review from an ecommerce site to give feedback to the Shipping deparmtment.
Summarize the review below, delimited by triple backticks, in at most 30 words, and focusing on any aspects that mention shipping and delivery of the product.
```

Try "extract" instead of "summarize", result is more brief and specific

## Inferring

``` prompt
What is the sentiment of the following product review?

# limit output
Give your answer as a signle word, either "positive", or "negative".

# multiple output
Identify a list of emotions that the writer of the following review is expressing. Include no more than 5 items in the list. Format your answer as a list of lower-case words separated by commas.

Is the writer of the following review expressing anger? Give your answer as either yes or no.

# format the output
Identify the following items from the review text:
- Item purchased by reviewer
- Company that made the item
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
Format you response as a JSON object with "Item", "Brand", "Sentiment" and "Anger" as the keys.
If the information isn't present, use "unknown" as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

# input a topic list and check if each topic is covered
topic_list = ['nasa', 'local government', 'engingeering', ...]
Determine whether each item in the following list of topics is a topic in the text below, which is delimited with triple backticks.
Give your answer as list with 0 or 1 for each topic.
List of topics: {", ".join(topic_list)}
Text sample: '''{story}'''
```

## Transforming

``` prompt
# translate
Translate the following English text to Spanish:

# identify language
Tell me which language this is:

# multiple languages translation
Translate the following  text to French and Spanish

Translate the following text to Spanish in both the formal and informal forms:
```

``` python
# batch processing multiple messages
for issue in user_messages:
    prompt = f"Tell me what language this is: ```{issue}```"
    lang = get_completion(prompt)
    print(f"Original message ({lang}): {issue}")

    prompt = f"""
    Translate the following  text to English \
    and Korean: ```{issue}```
    """
    response = get_completion(prompt)
    print(response, "\n")
```

``` prompt
# Tone transformation
Translate the following from slang to a business letter:

# Spellcheck/Grammar Check
proofread and correct this review:

Proofread and correct the following text and rewrite the corrected version. If you don't find and errors, just say "No errors found". Don't use any punctuation around the text:

proofread and correct this review. Make it more compelling.
Ensure it follows APA style guide and targets an advanced reader.
Output in markdown format.
```

Format conversion

``` python
data_json = { "resturant employees" :[
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""
response = get_completion(prompt)
```

``` python
text = f"""
"""
prompt = f"proofread and correct this review: ```{text}```"
response = get_completion(prompt)

# highlight differences in red lines
from redlines import Redlines

diff = Redlines(text, response)
display(Markdown(diff.output_markdown))
```

## Expanding

``` prompt
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ---,
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for their review.
If the sentiment is negative, apologize and suggest that they can reach out to customer service.
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as 'AI customer agent'.
Customer review : ---{review}---
Review sentiment: {sentiment}
```

Temperature: can be thought of as the degree of exploration or randomesss of the model.

- Temperature = 0, for tasks that require reliability, predictability
- Higher temperature (0.3, 0.7), for tasks that require variety

## Chatbot

[Jupyter Notebook](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/8/chatbot)

Roles:

- system: sets behavior of assistant
- assistant: chat mode
- user: you

``` python
# helper method for chatbot that takes multiple messages
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#   print(str(response.choices[0].message))
    return response.choices[0].message["content"]

messages =  [
{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},
{'role':'user', 'content':'tell me a joke'},
{'role':'assistant', 'content':'Why did the chicken cross the road'},
{'role':'user', 'content':'I don\'t know'}  ]

response = get_completion_from_messages(messages, temperature=1)

# example: OrderBot
def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context)
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))

    return pn.Column(*panels)

import panel as pn  # GUI
pn.extension()

panels = [] # collect display

context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ]  # accumulate messages

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

# show UI
dashboard

# summary
messages =  context.copy()
messages.append(
{'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
 The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},
)
#The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},

response = get_completion_from_messages(messages, temperature=0)
print(response)
```

## Prompts for Research

### Data Analysis

1. Data summary

    ``` prompt
    I would like you to help me with the initial data exploration of the customer satisfaction ratings dataset that I collected. Please provide insights on the following:
    1) Can you provide summary statistics of the customer satisfaction rating dataset, including the range, mean, median, and mode of the ratings?
    2) How many customers gave the highest rating (5) for each question, and how many gave the lowest rating (1)?

    Dataset: {}
    ```

2. Interpreting key insights

    ``` prompt
    Based on the above statistics, what are the key insights I can draw from the data? Can you also provide information about the following:
    1. Key drivers of customer loyalty based on available data?
    2. Common reasons for customer complaints or negative feedback?
    ```

### Research review

``` prompt
I need help with a literature review on {research_area}. Can you provide me with an overview of the current state of research in this area?

Can you provide a list of the top 10 most-cited papers on {research_area}?

Based on the current research, what are the main challenges and research gaps that need to be addressed in the {research_area}?

Can you suggest 5 important unanswered questions related to [your area of interest] that would advance the current state of knowledge in [specific subfield or topic]

Can you suggest the best research methodology and data collection techniques for studying [research topic] in [specific subfield or context], including their strengths, weaknesses, and when each would be most appropriate?

What are some effective strategies for developing a strong introduction, clear thesis statement, and convincing conclusion for my [research paper] on [research topic]? Please provide guiding questions and ideas on how to structure these elements to ensure they are effective and aligned with the research goals.

Proofread and edit my {Research Paper} for any grammatical, punctuation, repetitive words, and spelling errors. Please provide suggestions to improve the readability and flow of my research paper.

I would like you to generate a dataset of {Dataset About?} with {Number of Records} synthetic records with the following characteristics.
{Name of Field} ({Data Type / Range }) … and so on.
{Special Instructions about Dataset}
The data should be realistic and plausible, not obviously fake or randomly generated. Format the output as a comma-separated values (CSV) file with a header row listing the field names and {Dataset Number} data rows.
```

### ChatGPT limitations

- Accuracy and Reliability: ChatGPT is a language model, so the responses cannot be declared 100% accurate. You need to cross-verify your answers and consult additional sources.
- Ethics and Bias: Researchers should strive to maintain ethical research standards and be aware of potential biases in ChatGPT’s responses.

Ref: <https://machinelearningmastery.com/advanced-techniques-for-research-with-chatgpt/>
