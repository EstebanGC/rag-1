import logging
from typing import Dict, Any, Optional

from langchain.llms import ollama
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain.schema import Document

from config.settings import OLLAMA_CONFIG


