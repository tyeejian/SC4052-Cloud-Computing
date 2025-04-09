import streamlit as st
import requests
import json
import time
import base64
from urllib.parse import quote
import os
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
import torch
import threading

load_dotenv()


headers = {
    "Accept": "application/vnd.github.v3+json", 
    "X-GitHub-Api-Version": "2022-11-28",
    "Authorization" : f"Bearer {os.environ.get('GITHUB_TOKEN')}"
}

def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", 
        "content": """
        You are a helpful AI Assistant. Your task is to create a summary descriptions of a code function provided by the user and also generate the same functionality of code in another programming language as requested by the user.
        """},
    ]

    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move to model device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Create a streamer to print the output as it's generated
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    return messages, model, tokenizer, streamer

def generate_code_summary(prompt, messages, tokenizer, model, container):
    prompt += """
    In the summary, include the following:
    
    1. The programming language used in the code.
    2. Summary of the code functions, include function names and what it does.
    3. Line-by-line code breakdown.
    4. Suggest any improvements that can be made to the code.
    """

    # prompt += """
    # In the README, include the following:
    
    # 1. What is the project about
    # 2. How to build it
    # 3. How to run it
    # 4. Contact information
    # """
    
    messages.append({"role" : "user", "content" : prompt})
    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move to model device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
   # Set up streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Run generation in a background thread
    thread = threading.Thread(target=model.generate, kwargs={
        "inputs": model_inputs["input_ids"],
        "max_new_tokens": 512,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id,
    })
    thread.start()

    # Accumulate and display output in Streamlit container
    generated_text = ""
    for token in streamer:
        generated_text += token
        container.markdown(generated_text + "▌") 

    container.markdown(generated_text)


def search_github_repos(query, sort_by="stars", order="desc", per_page=10):
    """
    Search GitHub repositories based on a query.
    
    Parameters:
    - query: The search query
    - sort_by: How to sort results (stars, forks, updated)
    - order: Sort order (asc, desc)
    - per_page: Number of results per page (max 100)
    
    Returns:
    - Dictionary containing search results or error message
    """
    if not query:
        return {"error": "Please enter a search query"}
    
    base_url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": sort_by,
        "order": order,
        "per_page": per_page
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

def search_github_code(query, sort_by="best-match", order="desc", per_page=10):
    """
    Search GitHub code based on a query.
    
    Parameters:
    - query: The search query
    - sort_by: How to sort results (best-match, indexed)
    - order: Sort order (asc, desc)
    - per_page: Number of results per page (max 100)
    
    Returns:
    - Dictionary containing search results or error message
    """
    if not query:
        return {"error": "Please enter a search query"}
    
    base_url = "https://api.github.com/search/code"
    params = {
        "q": query,
        "sort": sort_by,
        "order": order,
        "per_page": per_page
    }
    
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

def get_file_content(url):
    """Get the content of a file from GitHub API"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.json()
        
        if content.get("encoding") == "base64" and content.get("content"):
            decoded_content = base64.b64decode(content["content"]).decode("utf-8")
            return decoded_content
        return "Content could not be decoded"
    except Exception as e:
        return f"Error retrieving content: {str(e)}"

def display_repo_info(repo):
    """Display relevant information about a repository"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### [{repo['full_name']}]({repo['html_url']})")
        st.markdown(f"**Description:** {repo['description']}" if repo['description'] else "**Description:** *No description available*")
        
        # Display programming language with color indicator
        if repo['language']:
            st.markdown(f"**Language:** {repo['language']}")
    
    with col2:
        st.metric("Stars", f"{repo['stargazers_count']:,}")
        st.metric("Forks", f"{repo['forks_count']:,}")
    
    st.markdown(f"**Last Updated:** {repo['updated_at'].split('T')[0]}")
    st.markdown("---")

def display_code_info(code_item):
    """Display information about a code search result"""
    repo_name = code_item["repository"]["full_name"]
    file_path = code_item["path"]
    file_url = code_item["html_url"]
    
    # Create a unique key for each expander
    expander_key = f"{repo_name}_{file_path}"
    
    st.markdown(f"### [{repo_name}/{file_path}]({file_url})")
    st.markdown(f"**Repository:** [{repo_name}]({code_item['repository']['html_url']})")
    
    with st.expander("View Code"):
        if code_item.get("content"):
            st.code(code_item["content"], language=file_path.split(".")[-1] if "." in file_path else "text")
        else:
            with st.spinner("Loading code..."):
                content = get_file_content(code_item["url"])
                st.code(content, language=file_path.split(".")[-1] if "." in file_path else "text")
    
    st.markdown("---")



if __name__ == "__main__":
    # Set up the Streamlit app
    st.set_page_config(page_title="GitHub Search Tool", page_icon="🔍", layout="wide")
    
    st.title("🔍 GitHub Search Tool")
    st.markdown("Search for repositories or code on GitHub based on keywords, programming languages, or topics.")

    # Create tabs for repository search and code search
    tab1, tab2, tab3 = st.tabs(["Repository Search", "Code Search", "LLM"])

    with tab1:
        st.header("Repository Search")
        
        # Repository search form
        with st.form(key="repo_search_form"):
            repo_search_query = st.text_input("Enter repository search query", 
                                        placeholder="Example: machine learning language:python stars:>1000")
            
            # Advanced search options
            col1, col2 = st.columns(2)
            with col1:
                repo_sort_option = st.selectbox("Sort by", ["stars", "forks", "updated"], key="repo_sort")
            with col2:
                repo_order_option = st.selectbox("Order", ["desc", "asc"], key="repo_order")
            
            repo_results_count = st.slider("Number of results", min_value=5, max_value=50, value=10, step=5, key="repo_count")
            
            repo_search_button = st.form_submit_button(label="Search GitHub Repositories")

        # Display help information for repository search
        with st.expander("Repository Search Tips"):
            st.markdown("""
            ### Advanced Repository Search Syntax:
            - **Keyword search:** `machine learning`
            - **Search by language:** `language:python`
            - **Search by stars:** `stars:>1000`
            - **Search by forks:** `forks:>500`
            - **Search by topic:** `topic:tensorflow`
            - **Search in name:** `in:name tensorflow`
            - **Search in description:** `in:description "machine learning framework"`
            - **Search by date:** `created:>2023-01-01`
            - **Combine multiple filters:** `machine learning language:python stars:>1000`
            """)

        # Handle repository search and display results
        if repo_search_button and repo_search_query:
            with st.spinner("Searching GitHub repositories..."):
                # Add slight delay to make spinner visible
                time.sleep(0.5)
                results = search_github_repos(repo_search_query, repo_sort_option, repo_order_option, repo_results_count)
            
            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"Found {results['total_count']:,} repositories. Showing top {min(repo_results_count, len(results['items']))}")
                
                for repo in results["items"]:
                    display_repo_info(repo)
                    
                if results["total_count"] > repo_results_count:
                    st.info(f"Showing {repo_results_count} of {results['total_count']:,} results. Refine your search to find more specific repositories.")
        elif repo_search_button and not repo_search_query:
            st.warning("Please enter a search query.")

    with tab2:
        st.header("Code Search")
        
        # Code search form
        with st.form(key="code_search_form"):
            code_search_query = st.text_input("Enter code search query", 
                                        placeholder="Example: function connect language:javascript")
            
            # Advanced search options
            col1, col2 = st.columns(2)
            with col1:
                code_sort_option = st.selectbox("Sort by", ["best-match", "indexed"], key="code_sort")
            with col2:
                code_order_option = st.selectbox("Order", ["desc", "asc"], key="code_order")
            
            code_results_count = st.slider("Number of results", min_value=5, max_value=30, value=10, step=5, key="code_count")
            
            code_search_button = st.form_submit_button(label="Search GitHub Code")

        # Display help information for code search
        with st.expander("Code Search Tips"):
            st.markdown("""
            ### Advanced Code Search Syntax:
            - **Keyword search:** `function parse`
            - **Search by language:** `language:javascript`
            - **Search by extension:** `extension:js`
            - **Search by filename:** `filename:config.json`
            - **Search by path:** `path:src/components`
            - **Search in specific repos:** `repo:facebook/react`
            - **Search by file size:** `size:<10000`
            - **Search by organization:** `org:microsoft`
            - **Combine multiple filters:** `function parse language:javascript extension:js`
            """)

        # Handle code search and display results
        if code_search_button and code_search_query:
            with st.spinner("Searching GitHub code..."):
                # Add slight delay to make spinner visible
                time.sleep(0.5)
                results = search_github_code(code_search_query, code_sort_option, code_order_option, code_results_count)
            
            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"Found {results['total_count']:,} code results. Showing top {min(code_results_count, len(results['items']))}")
                
                for code_item in results["items"]:
                    display_code_info(code_item)
                    
                if results["total_count"] > code_results_count:
                    st.info(f"Showing {code_results_count} of {results['total_count']:,} results. Refine your search to find more specific code.")
        elif code_search_button and not code_search_query:
            st.warning("Please enter a search query.")

    with tab3:
        st.header("LLM")
        st.write("Paste your code below for AI-Generated summary")
        # Load model and tokenizer
        if 'model_loaded' not in st.session_state:
            with st.spinner("Loading language model for code summarization..."):
                messages, model, tokenizer, streamer = load_model()
                if model and tokenizer:
                    st.session_state['model'] = model
                    st.session_state['tokenizer'] = tokenizer
                    st.session_state['messages'] = messages
                    st.session_state['streamer'] = streamer
                    st.session_state['model_loaded'] = True
                else:
                    st.error("Failed to load the language model. Please check the console for errors.")
        else:
            model = st.session_state['model']
            tokenizer = st.session_state['tokenizer']
            messages = st.session_state['messages']
            streamer = st.session_state['streamer']
        
        # Code input area
        code_input = st.text_area("Paste your code here:", height=300, 
                              placeholder="# Paste your code here")
        
        if st.button('Generate summary'):
            if not code_input or code_input == "# Paste your code here":
                st.warning("Please paste code")
            else:
                # Prepare UI elements for streaming output
                summary_placeholder = st.empty()
                summary_text = ""

                with st.spinner("Generating summary..."):
                    # Create a container for the streaming text
                    summary_container = st.empty()
                    generate_code_summary(code_input, messages, tokenizer, model,summary_container)

        
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and GitHub API. Note: GitHub API has rate limits for unauthenticated requests.")