
class BrowseNetQAPrompts:
    @staticmethod
    def one_shot_rag_qa_input_prompt():
        one_shot_rag_qa_docs = ('`Retrieved Context` :\n\n',
                                    'Wikipedia Title: Darlings (Kevin Drew album)\nContent: Darlings is the second solo album by Broken Social Scene co-founder Kevin Drew. It was released on March 18, 2014.',
                                    'Wikipedia Title: Kevin Drew\nContent: Kevin Drew (born September 9, 1976) is a Canadian musician and songwriter who, together with Brendan Canning, founded the expansive Toronto baroque-pop collective Broken Social Scene. He was also part of the lesser-known KC Accidental, which consisted of Drew and Charles Spearin, another current member of Broken Social Scene.',
                                    "Wikipedia Title: Philadelphia\nContent: Philadelphia is also a major hub for Greyhound Lines, which operates 24-hour service to points east of the Mississippi River. Most of Greyhound's services in Philadelphia operate to/from the Philadelphia Greyhound Terminal, located at 1001 Filbert Street in Center City Philadelphia. In 2006, the Philadelphia Greyhound Terminal was the second busiest Greyhound terminal in the United States, after the Port Authority Bus Terminal in New York. Besides Greyhound, six other bus operators provide service to the Center City Greyhound terminal: Bieber Tourways, Capitol Trailways, Martz Trailways, Peter Pan Bus Lines, Susquehanna Trailways, and the bus division for New Jersey Transit. Other services include Megabus and Bolt Bus.",
                                    'Wikipedia Title: Maiysha\nContent: Born in Minneapolis, Minnesota, United States, Maiysha went to Sarah Lawrence College where she studied vocal performance as well as creative writing and race and gender studies. After graduating, Maiysha moved to New York City to teach at a private school in Manhattan. She also was signed to Ford Models.',
                                    "Wikipedia Title: Toronto Coach Terminal\nContent: The Toronto Coach Terminal is the central bus station for inter-city services in Toronto, Ontario, Canada. It is located at 610 Bay Street, in the city's Downtown. The terminal is owned by Toronto Coach Terminal Inc., a wholly owned subsidiary of the Toronto Transit Commission (TTC). The TTC managed the station directly until July 8, 2012, when it was leased out in its entirety to bus lines Coach Canada and Greyhound Canada for $1.2 million annually. Opened in 1931 as the Gray Coach Terminal, the Art Deco style terminal was home base for Gray Coach, an interurban bus service then owned by the TTC. It replaced an earlier open air terminal, Gray Line Terminal."
                                    )
        
        one_shot_rag_qa_input_demo = (
            "\n\n`Question` : "
            "Where do greyhound buses leave from in the city where the performer of Darlings was born?"
            )
        one_shot_rag_qa_input_demo += (
            "\n\n`Subqueries` :\n 'Q1) Who is the performer of Darlings?  \nQ2) <Q1> In which city was the performer of Darlings born?  \nQ3) <Q2> Where do greyhound buses leave from in the city where the performer of Darlings was born?'")
        one_shot_rag_qa_input_demo += (
            f"\n\n{one_shot_rag_qa_docs}"
            '\nThought: '
        )

        return one_shot_rag_qa_input_demo

    @staticmethod
    def one_shot_rag_qa_output_prompt():
        one_shot_rag_qa_output_demo = (
            "Kevin Drew is the performer of Darlings. Kevin Drew was born in Toronto. Greyhound buses leave from Toronto Coach Terminal in Toronto, Ontario, Canada. \nAnswer: Toronto Coach Terminal (or) Toronto Coach Terminal, Toronto, Ontario, Canada."
        )
        return one_shot_rag_qa_output_demo

    @staticmethod
    def answer_generation_prompt():
        answer_generation_instruction = ('As an advanced reading comprehension assistant, your task is to analyze the retrieved context wikipedia passages to answer the Question meticulously. '
                                        'To arrive at the final answer, use the subqueries provided to you in the given order'
                                        'If you cannot answer a subquery, then try to answer the question to the best of your ability, based on the information available in the retrieved context.'
                                        'Start your response with "Thought: " where you systematically explain your reasoning process, breaking down the steps leading to the answer. '
                                        'Conclude with "Answer: " followed by a concise, definitive response—limited to just the essential words, no extra sentences or explanations. '
                                        'In case of multiple possible answers, provide answers with (or) in between, such as "American (or) America (or) United States of America".'
                                        'NOTE:'
                                        '1. I hope your answer matches the answer exactly, so ENSURE that the answer following "Answer:" is concise, such as 14 May, 1832  or yes. THE SHORTER, THE BETTER!!'
                                        '2. If the answer is a date, please provide the full date as much as possible, such as 18 May, 1932.'
                                        '3. If the answer is a place, please provide the full name of the place as much as possible, such as "Oxford, England" instead of "Oxford".')
        return answer_generation_instruction

class NaiveRAGQAPrompts:
    @staticmethod
    def one_shot_rag_qa_input_prompt():
        one_shot_rag_qa_docs = ('`Retrieved Context` :\n\n',
                                    'Wikipedia Title: Darlings (Kevin Drew album)\nContent: Darlings is the second solo album by Broken Social Scene co-founder Kevin Drew. It was released on March 18, 2014.',
                                    'Wikipedia Title: Kevin Drew\nContent: Kevin Drew (born September 9, 1976) is a Canadian musician and songwriter who, together with Brendan Canning, founded the expansive Toronto baroque-pop collective Broken Social Scene. He was also part of the lesser-known KC Accidental, which consisted of Drew and Charles Spearin, another current member of Broken Social Scene.',
                                    "Wikipedia Title: Philadelphia\nContent: Philadelphia is also a major hub for Greyhound Lines, which operates 24-hour service to points east of the Mississippi River. Most of Greyhound's services in Philadelphia operate to/from the Philadelphia Greyhound Terminal, located at 1001 Filbert Street in Center City Philadelphia. In 2006, the Philadelphia Greyhound Terminal was the second busiest Greyhound terminal in the United States, after the Port Authority Bus Terminal in New York. Besides Greyhound, six other bus operators provide service to the Center City Greyhound terminal: Bieber Tourways, Capitol Trailways, Martz Trailways, Peter Pan Bus Lines, Susquehanna Trailways, and the bus division for New Jersey Transit. Other services include Megabus and Bolt Bus.",
                                    'Wikipedia Title: Maiysha\nContent: Born in Minneapolis, Minnesota, United States, Maiysha went to Sarah Lawrence College where she studied vocal performance as well as creative writing and race and gender studies. After graduating, Maiysha moved to New York City to teach at a private school in Manhattan. She also was signed to Ford Models.',
                                    "Wikipedia Title: Toronto Coach Terminal\nContent: The Toronto Coach Terminal is the central bus station for inter-city services in Toronto, Ontario, Canada. It is located at 610 Bay Street, in the city's Downtown. The terminal is owned by Toronto Coach Terminal Inc., a wholly owned subsidiary of the Toronto Transit Commission (TTC). The TTC managed the station directly until July 8, 2012, when it was leased out in its entirety to bus lines Coach Canada and Greyhound Canada for $1.2 million annually. Opened in 1931 as the Gray Coach Terminal, the Art Deco style terminal was home base for Gray Coach, an interurban bus service then owned by the TTC. It replaced an earlier open air terminal, Gray Line Terminal."
                                    )
        
        one_shot_rag_qa_input_demo = (
            "\n\n`Question` : "
            "Where do greyhound buses leave from in the city where the performer of Darlings was born?"
            )
        one_shot_rag_qa_input_demo += (
            f"\n\n{one_shot_rag_qa_docs}"
            '\nThought: '
        )

        return one_shot_rag_qa_input_demo

    @staticmethod
    def one_shot_rag_qa_output_prompt():
        one_shot_rag_qa_output_demo = (
            "Kevin Drew is the performer of Darlings. Kevin Drew was born in Toronto. Greyhound buses leave from Toronto Coach Terminal in Toronto, Ontario, Canada. \nAnswer: Toronto Coach Terminal (or) Toronto Coach Terminal, Toronto, Ontario, Canada."
        )
        return one_shot_rag_qa_output_demo

    @staticmethod
    def answer_generation_prompt():
        answer_generation_instruction = ('You are an advanced reading comprehension assistant. Your goal is to analyze the given question and retrieved context carefully before generating an answer. '
                                        'Start your response with "Thought: " where you systematically explain your reasoning process, breaking down the steps leading to the answer. '
                                        'Conclude with "Answer: " followed by a concise, definitive response—limited to just the essential words, no extra sentences or explanations. '
                                        'In case of multiple possible answers, provide answers with (or) in between, such as "American (or) America (or) United States of America".'
                                        'NOTE:'
                                        '1. I hope your answer matches the answer exactly, so ENSURE that the answer following "Answer:" is concise, such as 14 May, 1832  or yes. THE SHORTER, THE BETTER!!'
                                        '2. If the answer is a date, please provide the full date as much as possible, such as 18 May, 1932.'
                                        '3. If the answer is a place, please provide the full name of the place as much as possible, such as "Oxford, England" instead of "Oxford".')

        return answer_generation_instruction