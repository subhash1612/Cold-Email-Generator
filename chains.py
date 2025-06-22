import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class OutputParserException(Exception):
    pass

class Chain:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = ChatPromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        llm_chain = prompt_extract | self.llm
        res = llm_chain.invoke(input={"page_data": cleaned_text})
        print(res)
        try:
            res = json.loads(res)
        except json.JSONDecodeError:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    def write_mail(self, job, links):
        # Format links as markdown bullet points
        if isinstance(links, list):
            formatted_links = ""
            for item in links:
                url = item.get("links") if isinstance(item, dict) else str(item)
                if url:
                    formatted_links += f"* {url}\n"
        else:
            formatted_links = str(links)

        prompt_email = ChatPromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are John, a business development executive at XYZ. XYZ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of XYZ 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase XYZ's portfolio:
            {link_list}
            Remember you are John, BDE at XYZ. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        llm_chain = prompt_email | self.llm
        res = llm_chain.invoke({"job_description": str(job), "link_list": formatted_links})
        return res
