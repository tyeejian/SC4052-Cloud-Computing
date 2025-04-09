# GitHub Search Tool
*This is generated using LLM, it has not beed vetted and read through thoroughly.*

This project is an assignment project in NTU for the module SC4052 - Cloud Computing in AY2024/25.


## Overview
This is a simple web application designed to facilitate the search for repositories or code using GitHub APIs. It allows users to perform basic searches within GitHub repositories and codes. The use of LLM is also available to generate code summary after searching for code in GitHub.

## Installation
To use this tool, you need to have the necessary permissions to access GitHub repositories and code snippets. If you do not have these permissions, you will need to obtain an API token through your GitHub account.

You can install the application using pip:

`pip install -r requirements.txt`

Once installed, you can start building your repositories or writing code snippets directly into the application.

## Usage
The application supports two main modes of operation:

1. **Repository Search**: Searches for repositories based on keyword queries, programming languages, or topics.
2. **Code Search**: Searches for code snippets based on keyword queries.
3. **LLM**: Generate code summary by copy-pasting code from **Code Search**.


## Key Features
- **Advanced Search Options**: Users can filter their search results using various criteria like sorting by stars, forks, or created dates.
- **Streamed Output**: Real-time summaries of the searched repositories or code snippets are displayed in a separate section.
- **User Interaction**: A user-friendly interface allows users to interactively manage their search results.

## User Interface
The application uses Streamlit, a lightweight Python library for creating web applications. It also leverages the `requests` library to fetch data from GitHub APIs.

## Future (possible) Improvments 
- One-click button to generate summary from the code search module
- Use Langchain to create a RAG to enable generation of README from all codes in the repository.
- Enable the use of different models


## Contributing (No need to contribute :))
Contributions to improve the application's functionality or enhance its user experience are welcome! To contribute, follow these steps:

1. Fork the repository (`git clone <owner>/<repository>`).
2. Make changes to the source files (`cd <directory> `and make necessary modifications).
3. Commit your changes (`git commit -m "Add feature [x]"`).
4. Push your commits to your forked repository (`git push origin master`).
After making your changes, submit a pull request (`git push origin main`). Your contributions will be reviewed and incorporated into the main branch.

Feel free to explore and experiment with the GitHub Search Tool to discover new ways to leverage GitHub APIs!