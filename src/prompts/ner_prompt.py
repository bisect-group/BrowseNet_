from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

def get_ner_prompt():
    
    one_shot_passage = """Radio City
    Radio City is India's first private FM radio station and was started on 3 July 2001.
    It plays Hindi, English and regional songs.
    Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

    one_shot_passage_entities = """{"named_entities":
        ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
    }
    """

    ner_instruction = """Your task is to extract named entities from the given paragraph. 
    Respond with a JSON list of entities.
    """

    ner_input_one_shot = """Paragraph:
    ```
    {}
    ```
    """.format(one_shot_passage)

    ner_output_one_shot = one_shot_passage_entities

    ner_user_input = "Paragraph:```\n{user_input}\n```"
    ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(ner_instruction),
                                                    HumanMessage(ner_input_one_shot),
                                                    AIMessage(ner_output_one_shot),
                                                    HumanMessagePromptTemplate.from_template(ner_user_input)])
    return ner_prompts