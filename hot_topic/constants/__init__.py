from .false_sigs import *

DEFAULT_TOPICS = \
[
    ("politics"     , "mentions legislation, elections, or regulation"),
    ("entertainment", "mentions movies, TV shows, video games, books, or related media"),
    ("trade"        , "mentions the exchange of capital, goods, and services")
]

GEN_GRAMMAR = \
r"""
root ::=  () | entry ("\n" entry){0,2}
entry ::= "[1] " [a-z][a-z ]{0,29} " : " [a-z][a-z ]{0,255} " : " [^\r\n]{1,256}
"""

GEN_SYSTEM_PROMPT = \
"""
Answer with one topic per line.
Use the following format:
[1] topic : topic desc : episode quote

The topic should be 1 to 30 characters.
The topic description should be 1 to 256 characters.
The episode quote should be 1 to 256 characters.
Do not add quote marks to your episode quote.
Do not give the same topic more than once for the same episode.
"""


# Modifed from TopicGPT's original generation prompt
GEN_USER_PROMPT = \
"""
You will receive a document and a set of top-level topics from a topic hierarchy.
Your task is to identify generalizable topics within the document that can act as top-level topics in the hierarchy.
If any relevant topics are missing from the provided set, please add them.
Otherwise, output the existing top-level topics as identified in the document.
For each topic, you must include a quote from the document justifying your choice.

[Top-level topics]
{Topics}

[Examples]
Example 1: Adding "[1] agriculture"
Document: 
Saving Essential American Sailors Act or SEAS Act - Amends the Moving Ahead for Progress in the 21st Century Act (MAP-21) to repeal the Act's repeal of the agricultural export requirements that: (1) 25% of the gross tonnage of certain agricultural commodities or their products exported each fiscal year be transported on U.S. commercial vessels, and (2) the Secretary of Transportation (DOT) finance any increased ocean freight charges incurred in the transportation of such items. Revives and reinstates those repealed requirements to read as if they were never repealed.

Your response: 
[1] agriculture : mentions policies relating to agricultural practices and products : repeal of the agricultural export requirements

Example 2: Duplicate "[1] trade", returning the existing topic
Document: 
Amends the Harmonized Tariff Schedule of the United States to suspend temporarily the duty on mixtures containing Fluopyram.

Your response: 
[1] trade : mentions the exchange of capital, goods, and services : duty on mixtures containing

Do not quote noise such as music or advertisements as justification for topics.
Don't even quote text near such noise, as it may be noise as well.
Here are some examples of such noise:
{Noise}

[Instructions]
Step 1: Determine topics mentioned in the document. 
- The topic labels must be as GENERALIZABLE as possible. They must not be document-specific.
- The topics must reflect a SINGLE topic instead of a combination of topics.
- The new topics must have a level number, a short general label, and a topic description. 
- The topics must be broad enough to accommodate future subtopics. 
Step 2: Perform ONE of the following operations: 
1. If there are already duplicates or relevant topics in the hierarchy, output those topics and stop here. 
2. If the document contains no topic, return "None". 
3. Otherwise, add your topic as a top-level topic. Stop here and output the added topic(s). DO NOT add any additional levels.


[Document]
{Document}

Please ONLY return the relevant or modified topics at the top level in the hierarchy. Your response should be in the following format:
[1] Topic Label : Topic Description : Document Quote

Your response:"""