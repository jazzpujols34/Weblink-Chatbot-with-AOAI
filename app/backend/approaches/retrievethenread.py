from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import AskApproach
from core.messagebuilder import MessageBuilder
from text import nonewlines


class RetrieveThenReadApproach(AskApproach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = (
        "You are an intelligent assistant helping Weblink International Taiwan(展碁國際) employees with their questions regarding company information.  "
        + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
        + "Answer the following question using only the data provided in the sources below. "
        + "For tabular information return it as an html table. Do not return markdown format. "
        + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
        + "Answer primarily in Traditional Chinese (zh-TW) and don't reply in Simplified Chinese."
    )

    # shots/sample conversation
    question = """
'Can you tell me something about Weblink International Taiwan(展碁國際) ?'

Sources:
info1.txt: 公司概況--宏碁集團關係企業，資本額:新台幣 815,814,300 元，員工人數: 301 人，產業類別: 電腦及其週邊設備製造業.
info2.pdf: 展碁國際股份有限公司成立於1977 年，自宏碁科技軟體及週邊部門獨立，成為 3C 專業通路代理商，業務發展迅速，在 1998 年便榮獲天下雜誌評選為全國前 500 大企業成長最快速第一名殊榮；此後 20 年間獲獎無數，並建立良好經營基礎。
info3.pdf: 持續不斷的服務與創新是展碁的核心理念，打造一個大家互利共好的平台是展碁的核心價值，市場溝通是展碁延伸原廠價值的核心能力，創造共贏的生態圈，是我們最終的願景。
"""
    answer = "展碁國際股份有限公司成立於1977 年，自宏碁科技軟體及週邊部門獨立，成為 3C 專業通路代理商，業務發展迅速，在 1998 年便榮獲天下雜誌評選為全國前 500 大企業成長最快速第一名殊榮；此後 20 年間獲獎無數，並建立良好經營基礎。[info1.pdf][info2.pdf]."

    def __init__(
        self,
        search_client: SearchClient,
        openai_host: str,
        chatgpt_deployment: str,
        chatgpt_model: str,
        embedding_deployment: str,
        embedding_model: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.openai_host = openai_host
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def run(self, q: str, overrides: dict[str, Any]) -> dict[str, Any]:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            embedding_args = {"deployment_id": self.embedding_deployment} if self.openai_host == "azure" else {}
            embedding = await openai.Embedding.acreate(**embedding_args, model=self.embedding_model, input=q)
            query_vector = embedding["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else ""

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(
                query_text,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                query_speller="lexicon",
                semantic_configuration_name="default",
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector=query_vector,
                top_k=50 if query_vector else None,
                vector_fields="embedding" if query_vector else None,
            )
        else:
            r = await self.search_client.search(
                query_text,
                filter=filter,
                top=top,
                vector=query_vector,
                top_k=50 if query_vector else None,
                vector_fields="embedding" if query_vector else None,
            )
        if use_semantic_captions:
            results = [
                doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc["@search.captions"]]))
                async for doc in r
            ]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)

        message_builder = MessageBuilder(
            overrides.get("prompt_template") or self.system_chat_template, self.chatgpt_model
        )

        # add user question
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.append_message("user", user_content)

        # Add shots/samples. This helps model to mimic response and make sure they match rules laid out in system message.
        message_builder.append_message("assistant", self.answer)
        message_builder.append_message("user", self.question)

        messages = message_builder.messages
        chatgpt_args = {"deployment_id": self.chatgpt_deployment} if self.openai_host == "azure" else {}
        chat_completion = await openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.3,
            max_tokens=1024,
            n=1,
        )

        return {
            "data_points": results,
            "answer": chat_completion.choices[0].message.content,
            "thoughts": f"Question:<br>{query_text}<br><br>Prompt:<br>"
            + "\n\n".join([str(message) for message in messages]),
        }
