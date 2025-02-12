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

## DeepSeek 使用技巧

## 做图

- SVG 矢量图
  - 基础图形
  - 图标
  - 简单插图
  - 流程图
  - 组织架构图
- Mermaid 图表
  - 流程图
  - 时序图
  - 类图
  - 状态图
  - 实体关系图
  - 思维导图
- React 图表
  - 折线图
  - 柱状图
  - 饼图
  - 散点图
  - 雷达图
  - 组合图表

### 推理模型 vs. 通用模型

#### 推理模型

- 优势领域: 数学推导、逻辑分析、代码生成、复杂问题拆解
- 劣势领域: 发散性任务（如诗歌创作）
- 性能本质: 专精于逻辑密度高的任务
- 强弱判断: 并非全面更强，仅在其训练目标领域显著优于通用模型
- 提示语更简洁，只需明确任务目标和需求（因其已内化推理逻辑）
- 无需逐步指导，模型自动生成结构化推理过程（若强行拆解步骤，反而可能限制其能力）

#### 通用模型

- 优势领域: 文本生成、创意写作、多轮对话、开放性问答
- 劣势领域: 需要严格逻辑链的任务（如数学证明）
- 性能本质: 擅长多样性高的任务
- 强弱判断: 通用场景更灵活，但专项任务需依赖提示语补偿能力
- 需显式引导推理步骤（如通过CoT提示），否则可能跳过关键逻辑
- 依赖提示语补偿能力短板（如要求分步思考、提供示例）

#### 关键原则

1. 模型选择: 优先根据任务类型而非模型热度选择（如数学任务选推理模型，创意任务选通用模型）
2. 提示语设计
   - 推理模型：简洁指令，聚焦目标，信任其内化能力。（“要什么直接说”）
   - 通用模型：结构化、 补偿性引导（“缺什么补什么”）
3. 避免误区
   - 不要对推理模型使用“启发式”提示（如角色扮演），可能干扰其逻辑主线。
   - 不要对通用模型“过度信任”（如直接询问复杂推理问题， 需分步验证结果）。

### 不同需求类型的适配策略

1. 决策需求: 需权衡选项、 评估风险、选择最优解
   - 需求表达公式: 目标 + 选项 + 评估标准
   - 推理模型适配策略: 要求逻辑推演和量化分析
   - 通用模型适配策略: 直接建议， 依赖模型经验归纳
2. 分析需求: 需深度理解数据/信息、发现模式或因果关系
   - 问题 + 数据/信息 + 分析方法
   - 推理模型: 触发因果链推导与假设验证
   - 通用模型: 表层总结或分类
3. 创造性需求: 需生成新颖内容（文本/设计/方案）
   - 主题 + 风格/约束 + 创新方向
   - 推理模型: 结合逻辑框架生成结构化创意
   - 通用模型: 自由发散， 依赖示例引导
4. 验证需求: 需检查逻辑自洽性、数据可靠性或方案可行性
   - 结论/方案 + 验证方法 + 风险点
   - 推理模型: 自主设计验证路径并排查矛盾
   - 通用模型: 简单确认， 缺乏深度推演
5. 执行需求: 需完成具体操作（代码/计算/流程）
   - 任务 + 步骤约束 + 输出格式
   - 推理模型: 自主优化步骤，兼顾效率与正确性
   - 通用模型: 严格按指令执行，无自主优化

### 提示语工程

#### 基本结构

- 指令(Instruction)：这是提示语的核心， 明确告诉AI你希望它执行什么任务。
- 上下文(Context)：为AI提供背景信息， 帮助它更准确地理解和执行任务。
- 期望(Expectation)：明确或隐含地表达你对AI输出的要求和预期。

### 提示语基本元素分类

- 信息类元素
  - 主题
  - 背景
  - 数据
  - 知识域
  - 参考
- 结构类元素
  - 格式
  - 结构
  - 风格
  - 长度
  - 可视化
- 控制类元素
  - 任务指令
  - 质量控制
  - 约束条件
  - 迭代指令
  - 输出验证

### 策略

1. 精准定义任务，减少模糊性 生成指令、去除多余信息
2. 适当分解复杂任务，降低AI认知负荷
3. 引入引导性问题，提升生成内容的深度, 设置多个层次的问题、促使AI对比或论证、 引导思维的多样性
4. 控制提示语长度，确保生成的准确性, 避免嵌套复杂的指令、保持简洁性、使用分步提示
5. 灵活运用开放式提示与封闭式提示
   - 开放式提示： 提出开放性问题， 允许AI根据多个角度进行生成
   - 封闭式提示： 提出具体问题或设定明确限制， 要求AI给出精准回答

幻觉生成陷阱： 当AI自信地胡说八道

应对策略：

- 明确不确定性： 鼓励AI在不确定时明确说明
- 事实核查提示： 要求AI区分已知事实和推测
- 多源验证： 要求AI从多个角度或来源验证信息
- 要求引用： 明确要求AI提供信息来源，便于验证

### 常用模型

提示语链CIRS模型（ Context, Instruction, Refinement, Synthesis）

SPECTRA模型（ Systematic Partitioning for Enhanced Cognitive Task Resolution in AI）

- Segmentation（ 分割） ： 将大任务分为独立但相关的部分
- Prioritization（ 优先级） ： 确定子任务的重要性和执行顺序
- Elaboration（ 细化） ： 深入探讨每个子任务的细节
- Connection（ 连接） ： 建立子任务之间的逻辑关联
- Temporal Arrangement（ 时序安排） ： 考虑任务的时间维度
- Resource Allocation（ 资源分配） ： 为每个子任务分配适当的注意力资源
- Adaptation（ 适应） ： 根据AI反馈动态调整任务结构

Geneplore模型（ Generate-Explore Model）

创造性思维包括两个主要阶段：生成阶段（ Generate） 和探索阶段（ Explore）

发散思维的提示语链设计（基于“ IDEA” 框架）

- Imagine（ 想象） ： 鼓励超越常规的思考
- Diverge（ 发散） ： 探索多个可能性
- Expand（ 扩展） ： 深化和拓展初始想法
- Alternate（ 替代） ： 寻找替代方案

聚合思维的提示语链设计 基于“ FOCUS” 框架

- Filter（ 筛选） ： 评估和选择最佳想法
- Optimize（ 优化） ： 改进选定的想法
- Combine（ 组合） ： 整合多个想法
- Unify（ 统一） ： 创建一致的叙述或解决方案
- Synthesize（ 综合） ： 形成最终结论

跨界思维的提示语链设计 基于“ BRIDGE” 框架

- Blend（ 混合） ： 融合不同领域的概念
- Reframe（ 重构） ： 用新视角看待问题
- Interconnect（ 互联） ： 建立领域间的联系
- Decontextualize（ 去情境化） ： 将概念从原始环境中抽离
- Generalize（ 泛化） ： 寻找普适原则
- Extrapolate（ 推演） ： 将原理应用到新领域

概念嫁接策略（ CGS） ： 创造性融合
知识转移技术（ KTT） ： 跨域智慧应用
随机组合机制（ RCM） ： 打破常规思维
极端假设策略（ EHS） ： 突破思维界限
多重约束策略（ MCS） ： 激发创造性问题解决
情感融入策略（ EIS） ： 增强文本感染力

TASTE框架

- Task (任务): 定义模型主要任务或生成内容。
- Audience (目标受众): 明确说明目标受众。
- Structure (结构): 为输出的内容提供明确的组织结构， 包括段落安排、 论点展开顺序或其他逻辑关系。
- Tone (语气): 指定模型回答时的语气或风格。
- Example (示例):例子或模板可帮助模型理解输出风格或格式。

ALIGN框架

- Aim (目标): 明确任务的最终目标。
- Level (难度级别): 定义输出的难度级别。
- Input (输入): 指定需要处理的输入数据或信息， 或要求模型依据某些事实或条件进行推理。
- Guidelines (指导原则): 提供模型在执行任务时应该遵循的规则或约束。
- Novelty (新颖性): 明确是否需要模型提供原创性、创新性的内容， 是否允许引用已有知识。

示例：

``` prompt
[系统指令] 你是一个具有自我反思能力的AI作家。 你的任务是创作一个短篇科幻故事， 同时生成对你创作过程的评论。 请遵循以下步骤：
（1） 创作一个500字左右的科幻短篇， 主题是“ 时间旅行的道德困境” 。
（2） 在每个关键情节点后， 插入一段括号内的自我反思， 解释：
    a. 你为什么选择这个情节发展
    b. 你考虑过哪些其他可能性
    c. 这个选择如何推动主题的探讨
（3） 在故事结束后， 提供一个200字左右的整体创作过程反思， 包括：
    a. 你遇到的主要创作挑战
    b. 你认为最成功和最需要改进的部分
    c. 如果重新创作， 你会做出什么不同的选择
请确保主要叙事和元叙事评论的语气有所区分， 以突出自反性特征。 开始你的创作。
```

``` prompt
[系统指令]你将扮演两个角色： 一个是小说家A， 另一个是评论家B。 你们将合作创作一篇关于"人工智能伦理"的文章。 遵循以下规则：
（1） 小说家A：
    a. 以小说的方式呈现“ 人工智能伦理” 的各个方面。
    b. 每写完约200字， 暂停让评论家B进行评论。
（2） 评论家B：
    a. 对A的写作进行简短的文学批评和伦理分析。
    b. 评论要简洁， 不超过50字。
（3） 互动规则：
    a. A在收到B的评论后， 必须在某种程度上采纳建议， 调整后续写作。
    b. 如果A不同意B的某个观点， 可以在后续写作中巧妙地反驳。
（4） 整体结构：
    a. 文章总长度控制在1000字左右。
    b. 以A的一段总结性反思结束全文。
请开始创作， 确保A和B的声音清晰可辨， 且整体形成一个连贯的叙事。
```

### 进阶使用

- 知识获取
  - 让AI列出关键概念清单
  - 通过AI提问激发思考
  - 用AI拓展思维维度
  - 概念图谱构建
    - AI辅助绘制知识地图
    - 识别知识关联和缺口
  - 深度学习对话
    - 主题式探讨
    - 多角度理解
  - 知识验证
    - 概念准确性验证
    - 理解深度测试

- 知识整合
  - AI协助建立知识关联
  - 发现知识应用场景
  - 形成系统化观点
  - 跨领域关联
    - 建立知识连接
    - 发现创新点
  - 系统化重构
    - 构建知识体系
    - 形成新框架
  - 情境化应用
    - 场景模拟
    - 实践验证

- 提示构建，创新突破
  - 整合关键信息要素
  - 构建清晰的结构
  - 设定具体的约束
  - 观点生成
    - 新观点构建
    - 创新性验证
  - 方法创新
    - 解决方案设计
    - 方法论构建
  - 价值创造
    - 实践应用
    - 价值验证

AI使用层次与突破路径

1. 基础使用层: 单一任务 / 简单提示词 / 被动应用
2. 进阶使用层: 任务组合 / 结构化提示词 / 主动优化
3. 创新使用层: 流程再造 / 提示词艺术 / 创造性应用

突破路径：

1. 建立提示词体系
2. 设计协作流程
3. 发展创新方法
4. 打造个人特色
