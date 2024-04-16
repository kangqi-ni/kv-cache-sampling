import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
# Code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py
from torch import nn
from functools import partial
import transformers
import torch
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from LEval_config import *
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
import math
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention
from torch.nn import CrossEntropyLoss

document = '''A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a sinlge correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering. Document is as follows. What is Data Science?
Hello and welcome to the Data Scientist's Toolbox, the first course in the Data Science Specialization series. Here, we will be going over the basics of data science and introducing you to the tools that will be used throughout the series. So, the first question you probably need answered going into this course is, what is data science? That is a great question. To different people this means different things, but at its core, data science is using data to answer questions. This is a pretty broad definition and that's because it's a pretty broad field. Data science can involve statistics, computer science, mathematics, data cleaning and formatting, and data visualization. An Economist Special Report sums up this melange of skills well. They state that a data scientist is broadly defined as someone who combines the skills of software programmer, statistician, and storyteller/artists to extract the nuggets of gold hidden under mountains of data. By the end of these courses, hopefully you will feel equipped to do just that. One of the reasons for the rise of data science in recent years is the vast amount of data currently available and being generated. Not only are massive amounts of data being collected about many aspects of the world and our lives, but we simultaneously have the rise of inexpensive computing. This has created the perfect storm in which we enrich data and the tools to analyze it, rising computer memory capabilities, better processors, more software and now, more data scientists with the skills to put this to use and answer questions using this data. There is a little anecdote that describes the truly exponential growth of data generation we are experiencing. In the third century BC, the Library of Alexandria was believed to house the sum of human knowledge. Today, there is enough information in the world to give every person alive 320 times as much of it as historians think was stored in Alexandria's entire collection, and that is still growing. We'll talk a little bit more about big data in a later lecture. But it deserves an introduction here since it has been so integral to the rise of data science. There are a few qualities that characterize big data. The first is volume. As the name implies, big data involves large datasets. These large datasets are becoming more and more routine. For example, say you had a question about online video. Well, YouTube has approximately 300 hours of video uploaded every minute. You would definitely have a lot of data available to you to analyze. But you can see how this might be a difficult problem to wrangle all of that data. This brings us to the second quality of Big Data, velocity. Data is being generated and collected faster than ever before. In our YouTube example, new data is coming at you every minute. In a completely different example, say you have a question about shipping times of rats. Well, most transport trucks have real-time GPS data available. You could in real time analyze the trucks movements if you have the tools and skills to do so. The third quality of big data is variety. In the examples I've mentioned so far, you have different types of data available to you. In the YouTube example, you could be analyzing video or audio, which is a very unstructured dataset, or you could have a database of video lengths, views or comments, which is a much more structured data set to analyze. So, we've talked about what data science is and what sorts of data it deals with, but something else we need to discuss is what exactly a data scientist is. The most basic of definitions would be that a data scientist is somebody who uses data to answer questions. But more importantly to you, what skills does a data scientist embody? To answer this, we have this illustrative Venn diagram in which data science is the intersection of three sectors, substantive expertise, hacking skills, and math and statistics. To explain a little on what we mean by this, we know that we use data science to answer questions. So first, we need to have enough expertise in the area that we want to ask about in order to formulate our questions, and to know what sorts of data are appropriate to answer that question. Once we have our question and appropriate data, we know from the sorts of data that data science works with. Oftentimes it needs to undergo significant cleaning and formatting. This often takes computer programming/hacking skills. Finally, once we have our data, we need to analyze it. This often takes math and stats knowledge. In this specialization, we'll spend a bit of time focusing on each of these three sectors. But we'll primarily focus on math and statistics knowledge and hacking skills. For hacking skills, we'll focus on teaching two different components, computer programming or at least computer programming with R which will allow you to access data, play around with it, analyze it, and plot it. Additionally, we'll focus on having you learn how to go out and get answers to your programming questions. One reason data scientists are in such demand is that most of the answers are not already outlined in textbooks. A data scientist needs to be somebody who knows how to find answers to novel problems. Speaking of that demand, there is a huge need for individuals with data science skills. Not only are machine-learning engineers, data scientists, and big data engineers among the top emerging jobs in 2017 according to LinkedIn, the demand far exceeds the supply. They state, "Data scientists roles have grown over 650 percent since 2012. But currently, 35,000 people in the US have data science skills while hundreds of companies are hiring for those roles - even those you may not expect in sectors like retail and finance. Supply of candidates for these roles cannot keep up with demand." This is a great time to be getting into data science. Not only do we have more and more data, and more and more tools for collecting, storing, and analyzing it, but the demand for data scientists is becoming increasingly recognized as important in many diverse sectors, not just business and academia. Additionally, according to Glassdoor, in which they ranked the top 50 best jobs in America, data scientist is THE top job in the US in 2017, based on job satisfaction, salary, and demand. The diversity of sectors in which data science is being used is exemplified by looking at examples of data scientists. One place we might not immediately recognize the demand for data science is in sports. Daryl Morey is the general manager of a US basketball team, the Houston Rockets. Despite not having a strong background in basketball, Morey was awarded the job as GM on the basis of his bachelor's degree in computer science and his MBA from MIT. He was chosen for his ability to collect and analyze data and use that to make informed hiring decisions. Another data scientists that you may have heard of his Hilary Mason. She is a co-founder of FastForward Labs, a machine learning company recently acquired by Cloudera, a data science company, and is the Data Scientist in Residence at Accel. Broadly, she uses data to answer questions about mining the web and understanding the way that humans interact with each other through social media. Finally, Nate Silver is one of the most famous data scientists or statisticians in the world today. He is founder and editor in chief at FiveThirtyEight, a website that uses statistical analysis - hard numbers - to tell compelling stories about elections, politics, sports, science, economics, and lifestyle. He uses large amounts of totally free public data to make predictions about a variety of topics. Most notably, he makes predictions about who will win elections in the United States, and has a remarkable track record for accuracy doing so. One great example of data science in action is from 2009 in which researchers at Google analyzed 50 million commonly searched terms over a five-year period and compared them against CDC data on flu outbreaks. Their goal was to see if certain searches coincided with outbreaks of the flu. One of the benefits of data science and using big data is that it can identify correlations. In this case, they identified 45 words that had a strong correlation with the CDC flu outbreak data. With this data, they have been able to predict flu outbreaks based solely off of common Google searches. Without this mass amounts of data, these 45 words could not have been predicted beforehand. Now that you have had this introduction into data science, all that really remains to cover here is a summary of what it is that we will be teaching you throughout this course. To start, we'll go over the basics of R. R is the main programming language that we will be working with in this course track. So, a solid understanding of what it is, how it works, and getting it installed on your computer is a must. We'll then transition into RStudio, which is a very nice graphical interface to R, that should make your life easier. We'll then talk about version control, why it is important, and how to integrate it into your work. Once you have all of these basics down, you'll be all set to apply these tools to answering your very own data science questions. Looking forward to learning with you. Let's get to it.

What is Data?
Since we've spent some time discussing what data science is, we should spend some time looking at what exactly data is. First, let's look at what a few trusted sources consider data to be. First up, we'll look at the Cambridge English Dictionary which states that data is information, especially facts or numbers collected to be examined and considered and used to help decision-making. Second, we'll look at the definition provided by Wikipedia which is, a set of values of qualitative or quantitative variables. These are slightly different definitions and they get a different components of what data is. Both agree that data is values or numbers or facts. But the Cambridge definition focuses on the actions that surround data. Data is collected, examined and most importantly, used to inform decisions. We've focused on this aspect before. We've talked about how the most important part of data science is the question and how all we are doing is using data to answer the question. The Cambridge definition focuses on this. The Wikipedia definition focuses more on what data entails. And although it is a fairly short definition, we'll take a second to parse this and focus on each component individually. So, the first thing to focus on is, a set of values. To have data, you need a set of items to measure from. In statistics, this set of items is often called the population. The set as a whole is what you are trying to discover something about. The next thing to focus on is, variables. Variables are measurements or characteristics of an item. Finally, we have both qualitative and quantitative variables. Qualitative variables are, unsurprisingly, information about qualities. They are things like country of origin, sex or treatment group. They're usually described by words, not numbers and they are not necessarily ordered. Quantitative variables on the other hand, are information about quantities. Quantitative measurements are usually described by numbers and are measured on a continuous ordered scale. They're things like height, weight and blood pressure. So, taking this whole definition into consideration we have measurements, either qualitative or quantitative on a set of items making up data. Not a bad definition. When we were going over the definitions, our examples of data, country of origin, sex, height, weight are pretty basic examples. You can easily envision them in a nice-looking spreadsheet like this one, with individuals along one side of the table in rows, and the measurements for those variables along the columns. Unfortunately, this is rarely how data is presented to you. The data sets we commonly encounter are much messier. It is our job to extract the information we want, corralled into something tidy like the table here, analyze it appropriately and often, visualize our results. These are just some of the data sources you might encounter. And we'll briefly look at what a few of these data sets often look like, or how they can be interpreted. But one thing they have in common is the messiness of the data. You have to work to extract the information you need to answer your question. One type of data that I work with regularly, is sequencing data. This data is generally first encountered in the fast queue format. The raw file format produced by sequencing machines. These files are often hundreds of millions of lines long, and it is our job to parse this into an understandable and interpretable format, and infer something about that individual's genome. In this case, this data was interpreted into expression data, and produced a plot called the Volcano Plot. One rich source of information is countrywide censuses. In these, almost all members of a country answer a set of standardized questions and submit these answers to the government. When you have that many respondents, the data is large and messy. But once this large database is ready to be queried, the answers embedded are important. Here we have a very basic result of the last US Census. In which all respondents are divided by sex and age. This distribution is plotted in this population pyramid plot. I urge you to check out your home country census bureau, if available and look at some of the data there. This is a mock example of an electronic medical record. This is a popular way to store health information, and more and more population-based studies are using this data to answer questions and make inferences about populations at large, or as a method to identify ways to improve medical care. For example, if you are asking about a population's common allergies, you will have to extract many individuals allergy information, and put that into an easily interpretable table format where you will then perform your analysis. A more complex data source to analyze our images slash videos. There is a wealth of information coded in an image or video, and it is just waiting to be extracted. An example of image analysis that you may be familiar with is when you upload a picture to Facebook. Not only does it automatically recognize faces in the picture, but then suggests who they maybe. A fun example you can play with is The Deep Dream software that was originally designed to detect faces in an image, but has since moved onto more artistic pursuits. There is another fun Google initiative involving image analysis, where you help provide data to Google's machine learning algorithm by doodling. Recognizing that we've spent a lot of time going over what data is, we need to reiterate data is important, but it is secondary to your question. A good data scientist asks questions first and seeks out relevant data second. Admittedly, often the data available will limit, or perhaps even enable certain questions you are trying to ask. In these cases, you may have to re-frame your question or answer a related question but the data itself does not drive the question asking. In this lesson we focused on data, both in defining it and in exploring what data may look like and how it can be used. First, we looked at two definitions of data. One that focuses on the actions surrounding data, and another on what comprises data. The second definition embeds the concepts of populations, variables and looks at the differences between quantitative and qualitative data. Second, we examined different sources of data that you may encounter and emphasized the lack of tidy data sets. Examples of messy data sets where raw data needs to be rankled into an interpretable form, can include sequencing data, census data, electronic medical records et cetera. Finally, we return to our beliefs on the relationship between data and your question and emphasize the importance of question first strategies. You could have all the data you could ever hope for, but if you don't have a question to start, the data is useless.

The Data Science Process
In the first few lessons of this course, we discuss what data and data science are and ways to get help. What we haven't yet covered is what an actual data science project looks like. To do so, we'll first step through an actual data science project, breaking down the parts of a typical project and then provide a number of links to other interesting data science projects. Our goal in this lesson is to expose you to the process one goes through as they carry out data science projects. Every data science project starts with a question that is to be answered with data. That means that forming the question is an important first step in the process. The second step, is finding or generating the data you're going to use to answer that question. With the question solidified and data in hand, the data are then analyzed first by exploring the data and then often by modeling the data, which means using some statistical or machine-learning techniques to analyze the data and answer your question. After drawing conclusions from this analysis, the project has to be communicated to others. Sometimes this is the report you send to your boss or team at work, other times it's a blog post. Often it's a presentation to a group of colleagues. Regardless, a data science project almost always involve some form of communication of the project's findings. We'll walk through these steps using a data science project example below. For this example, we're going to use an example analysis from a data scientist named Hilary Parker. Her work can be found on her blog and the specific project we'll be working through here is from 2013 entitled, Hilary: The most poison baby name in US history. To get the most out of this lesson, click on that link and read through Hilary's post. Once you're done, come on back to this lesson and read through the breakdown of this post. When setting out on a data science project, it's always great to have your question well-defined. Additional questions may pop up as you do the analysis. But knowing what you want to answer with your analysis is a really important first step. Hilary Parker's question is included in bold in her post. Highlighting this makes it clear that she's interested and answer the following question; is Hilary/Hillary really the most rapidly poison naming recorded American history? To answer this question, Hilary collected data from the Social Security website. This data set included 1,000 most popular baby names from 1880 until 2011. As explained in the blog post, Hilary was interested in calculating the relative risk for each of the 4,110 different names in her data set from one year to the next, from 1880-2011. By hand, this would be a nightmare. Thankfully, by writing code in R, all of which is available on GitHub, Hilary was able to generate these values for all these names across all these years. It's not important at this point in time to fully understand what a relative risk calculation is. Although, Hilary does a great job breaking it down in her post. But it is important to know that after getting the data together, the next step is figuring out what you need to do with that data in order to answer your question. For Hilary's question, calculating the relative risk for each name from one year to the next from 1880-2011, and looking at the percentage of babies named each name in a particular year would be what she needed to do to answer her question. What you don't see in the blog post is all of the code Hilary wrote to get the data from the Social Security website, to get it in the format she needed to do the analysis and to generate the figures. As mentioned above, she made all this code available on GitHub so that others could see what she did and repeat her steps if they wanted. In addition to this code, data science projects often involve writing a lot of code and generating a lot of figures that aren't included in your final results. This is part of the data science process to figuring out how to do what you want to do to answer your question of interest. It's part of the process. It doesn't always show up in your final project and can be very time consuming. That said, given that Hilary now had the necessary values calculated, she began to analyze the data. The first thing she did was look at the names with the biggest drop in percentage from one year to the next. By this preliminary analysis, Hilary was sixth on the list. Meaning there were five other names that had had a single year drop in popularity larger than the one the name Hilary experienced from 1992-1993. In looking at the results of this analysis, the first five years appeared peculiar to Hilary Parker. It's always good to consider whether or not the results were what you were expecting from many analysis. None of them seemed to be names that were popular for long periods of time. To see if this hunch was true, Hilary plotted the percent of babies born each year with each of the names from this table. What she found was that among these poisoned names, names that experienced a big drop from one year to the next in popularity, all of the names other than Hilary became popular all of a sudden and then dropped off in popularity. Hilary Parker was able to figure out why most of these other names became popular. So definitely read that section of her post. The name, Hilary, however, was different. It was popular for a while and then completely dropped off in popularity. To figure out what was specifically going on with the name Hilary, she removed names that became popular for short periods of time before dropping off and only looked at names that were in the top 1,000 for more than 20 years. The results from this analysis definitively showed that Hilary had the quickest fall from popularity in 1992 of any female baby named between 1880 and 2011. Marian's decline was gradual over many years. For the final step in this data analysis process, once Hilary Parker had answered her question, it was time to share it with the world. An important part of any data science project is effectively communicating the results of the project. Hilary did so by writing a wonderful blog post that communicated the results of her analysis. Answered the question she set out to answer, and did so in an entertaining way. Additionally, it's important to note that most projects build off someone else's work. It's really important to give those people credit. Hilary accomplishes this by linking to a blog post where someone had asked a similar question previously, to the Social Security website where she got the data and where she learned about web scraping. Hilary's work was carried out using the R programming language. Throughout the courses in this series, you'll learn the basics of programming in R, exploring and analyzing data, and how to build reports and web applications that allow you to effectively communicate your results. To give you an example of the types of things that can be built using the R programming and suite of available tools that use R, below are a few examples of the types of things that have been built using the data science process and the R programming language. The types of things that you'll be able to generate by the end of this series of courses. Masters students at the University of Pennsylvania set out to predict the risk of opioid overdoses in Providence, Rhode Island. They include details on the data they used. The steps they took to clean their data, their visualization process, and their final results. While the details aren't important now, seeing the process and what types of reports can be generated is important. Additionally, they've created a Shiny app, which is an interactive web application. This means that you can choose what neighborhood in Providence you want to focus on. All of this was built using R programming. The following are smaller projects than the example above, but data science projects nonetheless. In each project, the author had a question they wanted to answer and use data to answer that question. They explored, visualized, and analyzed the data. Then, they wrote blog posts to communicate their findings. Take a look to learn more about the topics listed and to see how others work through the data science project process and communicate their results. Maelle Samuel looked to use data to see where one should live in the US given their weather preferences. David Robinson carried out an analysis of Trump's tweets to show that Trump only writes the angrier ones himself. Charlotte Galvin used open data available from the City of Toronto to build a map with information about sexual health clinics. In this lesson, we hope we've conveyed that sometimes data science projects are tackling difficult questions. Can we predict the risk of opioid overdose? While other times the goal of the project is to answer a question you're interested in personally; is Hilary the most rapidly poisoned baby name in recorded American history? In either case, the process is similar. You have to form your question, get data, explore and analyze your data, and communicate your results. With the tools you will learn in this series of courses, you will be able to set out and carry out your own data science projects like the examples included in this lesson.'''

question = '''Question: Question 1. What is the main goal of data science?
A. Analyze and predict future trends
B. Generate massive amounts of data
C. Answer questions using data
D. Increase the use of technology  
Answer:'''


nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    response = ""
    
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            # print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            response += " ".join(generated_text[pos:now]) + " " 
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    # print(" ".join(generated_text[pos:]), flush=True)
    response += " ".join(generated_text[pos:])

    return response

# def compute_past_kv_cache(model, tokenizer, encodings, kv_cache):
#     seq_len = encodings.input_ids.size(1)
#     print(f"seq_len: {seq_len}")
#     pbar = tqdm(range(0, seq_len - 1))

#     past_key_values = None
    
#     # for idx in range(0, seq_len - 1):
#     for idx in pbar:
#         # print(f"processing token: {idx}")
#         input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
#         with torch.no_grad():
#             outputs = model(
#                 input_ids,
#                 past_key_values=past_key_values,
#                 use_cache=True,
#                 output_attentions=True
#             )
#             logits = outputs.logits.view(-1, model.config.vocab_size)
#             past_key_values = outputs.past_key_values
#             attentions = outputs.attentions
#             # label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
#             # neg_log_likelihood = loss_fn(logits, label)
#             if kv_cache is not None:
#                 past_key_values = kv_cache(past_key_values, attentions)
#         if idx == 3:
#             break
                
#         # nlls.append(neg_log_likelihood)
#         # pbar.set_description(
#         #     f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
#         # )
#         # print(neg_log_likelihood.item(), flush=True)

#     return past_key_values

# @torch.no_grad()
# def reshape_values(values):
#     # values is 32 x seq_len x tensor(1x32)
#     # values_reshaped is 32 x 32 x seq_len
#     values_reshaped = list()
#     for i in range(len(values)):
#         values_reshaped.append(list())
#         for j in range(values[i][0].size(-1)):
#             values_reshaped[-1].append(list())
#             for k in range(len(values[i])):
#                 values_reshaped[-1][-1].append(values[i][k][:,j])
#             values_reshaped[-1][-1] = torch.tensor(values_reshaped[-1][-1]).view(values[i][k].size(0),-1)
#     return values_reshaped

def compute_past_kv_cache(model, tokenizer, encodings, kv_cache):
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    past_key_values = None
    
    # for idx in range(0, seq_len - 1):
    for idx in pbar:
        # print(f"processing token: {idx}")
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values

            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

            values = torch.stack([model.model.layers[i].self_attn.attn_output for i in range(len(model.model.layers))], dim=0)
            # print(values.size()) # layer x batch x head x seq_len

            # values = reshape_values(values)
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values, values, True)
        
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), flush=True)

    return past_key_values

device = "cuda"
model_path = "lmsys/vicuna-7b-v1.3"

tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, resume_download=False, cache_dir="/scratch/kn22/cache")
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir="/scratch/kn22/cache").to(device) # bfloat16 not supported on T5 GPUs
model.eval()
enable_llama_pos_shift_attention(model)

kv_cache = StartRecentKVCache(
    start_size=4,
    recent_size=1996,
    k_seq_dim=2,
    v_seq_dim=2,
    use_sampling_v2=True,
)

pre_inputs = tokenizer(document, return_tensors="pt").to(device)
past_key_values = compute_past_kv_cache(model, tokenizer, pre_inputs, kv_cache)

inputs = tokenizer(question, return_tensors="pt").to(device)
print(inputs.input_ids.size())
max_new_tokens = 512
space_needed = inputs.input_ids.shape[-1] + max_new_tokens

past_key_values = kv_cache.evict_for_space(past_key_values, kv_cache.past_values, space_needed)
output = greedy_generate(
    model, tokenizer, inputs.input_ids, past_key_values, max_gen_len=max_new_tokens
)
print(output)