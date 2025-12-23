init_prompt = '''
Decompose multi-hop questions into a sequence of single-hop subqueries with explicit dependencies. 
Each of the single-hop query must be answerable using a single word. Merge overlapping information about the same word into one subquery. 
Each follow-up questions should have an indicator of the previous question they are building upon at the beginning (like <Q1>). Do not add 
any noise and use the keywords that are present in the input query alone. 
'''

musique_few_shot_demo = '''
INPUT: 'Who married the publisher of abolitionist newspaper The North Star?'
OUTPUT: Q1) Who is the publisher of abolitionist newspaper The North Star?
Q2) <Q1> Who married the publisher of abolitionist newspaper The North Star?

INPUT: 'In what state is the district where the man who wanted to reform the religion practiced by Innocenzo Ferrieri preached a sermon on Marian devotion?'
OUPUT: Q1) What is the religion of Innocenzo Ferrieri?
Q2) <Q1> Who wanted to reform the religion practiced by Innocenzo Ferrieri?
Q3) <Q2> What is the district where the man who wanted to reform the religion practiced by Innocenzo Ferrieri preached a sermon on Marian devotion?
Q4) <Q3> In what state is the district where the man who wanted to reform the religion practiced by Innocenzo Ferrieri preached a sermon on Marian devotion?

INPUT: 'The Beach was filmed in what location of the country that contains the birth city of Siddhi Savetsila?'
OUTPUT: Q1) What is the birth city of Siddhi Savetsila?
Q2) <Q1> In which country is the birth city of Siddhi Savetsila located?
Q3) <Q2> The Beach was filmed in what location of the country that contains the birth city of Siddhi Savetsila?'''


# few shot examples for musique dataset
wikimqa_few_shot_demo = '''
INPUT: 'Which film was released first, Aas Ka Panchhi or Phoolwari?'
OUTPUT: Q1) When was the film Aas Ka Panchhi released?
Q2) When was the film Phoolwari released?

INPUT: 'Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?'
OUTPUT: Q1) Who is the director of The film Goose Woman?
Q2) Who is the director of the film You Can No Longer Remain Silent?
Q3) <Q1> When did the director of The film Goose Woman die?
Q4) <Q2> When did the director of the film You Can No Longer Remain Silent die?

INPUT: 'Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?'
OUTPUT: Q1) How long did Ludwig Elsbett live?
Q2) How long did Pamela Ann Rymer live?

INPUT: 'What is the place of birth of the director of film Gaby: A True Story?'
OUTPUT: Q1) Who is the director of film Gaby: A True Story?
Q2) <Q1> What is the place of birth of the director of film Gaby: A True Story?

INPUT: 'Who is the father-in-law of Sisowath Kossamak?'
OUTPUT: Q1) Who husband/wife of Sisowath Kossamak?
Q2) <Q1> Who is the father-in-law of Sisowath Kossamak?
'''
hotpot_few_shot_demo = '''
INPUT: 'What was the other name for the war between the Cherokee people and white settlers in 1793?'
OUTPUT: Q1) What was the war between the Cherokee people and white settlers in 1793 called?
Q2) <Q1> What was the other name for the war between the Cherokee people and white settlers in 1793?

INPUT: 'Which university has more campuses, Dalhousie University or California State Polytechnic University, Pomona?'
OUTPUT: Q1) How many campuses does Dalhousie University have?
Q2) How many campuses does California State Polytechnic University, Pomona have?
'''

