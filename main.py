import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from secret_key import huggingface_api_key
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint


df = pd.read_pickle('data.pkl')
similarity_matrix = cosine_similarity(list(df['vector']))
huggingface_api_key = huggingface_api_key
#creating an instance of the HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2" , huggingfacehub_api_token=huggingface_api_key)



def recommend_based_on_history(article_indices, top_n=5):
    mean_similarity_scores = np.mean([similarity_matrix[i] for i in article_indices], axis=0)
    sorted_indices = np.argsort(mean_similarity_scores)[::-1]
    recommended_indices = [i for i in sorted_indices if i not in article_indices][:top_n]
    return recommended_indices

def summary(article,llm):
    prompt_tempate = PromptTemplate(
    input_variables = ['article'],
    template = "<s>[INST]You are an expert at making strong factual summarizations.\
                Take the article submitted by the user and produce a factual useful summary in around 100  words  :  {article}[/INST]"
)   
    
    summary = llm.invoke(prompt_tempate.format(article = article))
    return summary




#these are the article indices of last 5 watched ariticle 
last_watched_articles = [0, 1, 1000, 3, 4] 
recommended_articles = recommend_based_on_history(last_watched_articles)


#for summary implementation
article =  "EIN BOKEK, Israel (Reuters) - The Dead Sea is shrinking at the rate of about a meter a year, leaving behind deserted beaches and sinkholes in a slow-motion environmental disaster.   The main culprit is the drying up of the Jordan river, its main tributary, as communities upstream draw on it for farming and drinking. But mineral extraction makes the crisis worse - of the 700-800 million cubic meters of water lost each year, 250-350 million cubic meters is due to mining, Israel estimates.  Up to now, the Israeli government has rarely intervened in the operations of the biggest extractor: the Dead Sea Works, formerly state-owned and now operated under a 70-year concession by Israel Chemicals (ICL).  That is about to change.   Israel wants to re-tender the Dead Sea mining concession as much as eight years ahead of schedule, in 2022. It is motivated not only by environmental concerns but also by worries ICL will hold off on new investments in the concession’s final years.  The government believes ICL will agree to its proposal, first because the firm will have the right of first refusal but also because it too has a powerful reason to scrap the current concession: an article that gives the government the rights to interfere in investments starting in 2020.  The plant is one of ICL’s core assets, producing potash that goes into fertilizers, bromine for flame retardants and other products sold for billions of dollars worldwide.  The company, controlled by billionaire Idan Ofer’s Israel Corp, has not made its position clear. It declined to give an immediate comment on its stance when contacted by Reuters.  “This is a one-time opportunity, as the concession comes to an end and we enter a new period, to set standards for the factory’s operations and the environmental impact on the whole area,” said Galit Cohen, deputy director-general for policy and planning at the Environmental Protection Ministry.  Cohen was on the high-ranking inter-ministerial committee that produced a preliminary report in May with guidelines that aim to balance profits with environmental interests in the Dead Sea for the first time.    At the moment, ICL is largely free to do whatever it wants to maximize production, Cohen said, speaking to Reuters underneath a date tree on a northern beach at the lake.  “They have no incentive to reduce the amount of water they pump or think about from where they get the earth to build their dikes,” she said.   The Dead Sea has been popular for millennia for health seekers and tourists who come to float in its high-density waters and smear its mud on their skin. Without intervention, it will keep losing water, essential to the mineral extraction process, though experts believe it may eventually reach equilibrium at a much smaller size.   ICL said in a July 5 letter to the committee that its report raised “complicated legal, economic, operational and engineering issues, and ICL has significant reservations about part of what was said in it”.  “The company is studying the report and will relate to it as customary within the framework of the public hearing,” ICL said in a statement to Reuters.  In its 2017 annual report, the company said its ability to refinance debt in the next decade “... depends, among other things, on extension of the concession beyond 2030.”   The factory’s new license, whose term has not been set, will include pumping limits coupled with financial incentives to use less water, the committee’s report said. The amount of territory open to quarrying and drilling for wells will be reduced.       Final recommendations due around September are not expected to differ materially from the interim report’s, said a senior government official, who asked not to be identified given the sensitivity of the issue.  “We think everyone has an interest in making the tender earlier,” the official said. “The value of the asset gets lower as we get closer to the end of the concession period and it’s unclear what will happen after 2030.”  When the company was privatized in the 1990s, the government kept a “golden share” that gave it some oversight, in addition to the obligation under the terms of the concession that the company seek its approval for any new investment.  Michael Vatine, an analyst with Halman-Aldubi Investment House, said ICL was likely to want to avoid a decade of close government scrutiny.  “I think the company understands it needs to clear the fog regarding the long-term ... and not leave its investors feeling uncertain,” he said.  With revenue of $5.4 billion in 2017, ICL manufactures a range of products from industrial chemicals to food additives. It is the world’s sixth-largest producer of potash and supplies about a third of the world’s bromine, used in fire retardants.  The company does not share publicly how much of its revenues come from the Dead Sea, where it also mines magnesium and salts.  Costs at the factory are lower than at conventional mines, which are often hundreds of meters deep. Solar evaporation is less energy intensive and the climate allows mountains of potash to be stored outside and sold when prices are high.   According to their annual reports, ICL produced 3.7 million tonnes of potash at the Dead Sea in 2017 vs 2.1 million tonnes extracted by Arab Potash, which has exclusive rights on the Jordanian side that expire in 2058.  As ICL describes it, there is a virtually unlimited supply.  Bidders in a new tender would likely include the usual suspects from the small number of leading potash producers,  including Russia’s Uralkali, Germany’s K+S AG and Canada’s Nutrien, the report said.  Committee chair Yoel Naveh said it was possible competitors would be scared off by ICL’s numerous advantages: not just right of first refusal but also its deep knowledge of the project.  “The state needs to set a price and below that not give it to a private concessionaire,” he told parliament in June.  If the minimum failed to be met, the state should take over, he said. If someone else won, ICL would be compensated, he said, without naming a figure.  Editing by Sonya Hepinstall"
summary = summary(article,llm)

