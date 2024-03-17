import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { Ollama } from "@langchain/community/llms/ollama";
import { PromptTemplate } from "@langchain/core/prompts";

const llm = new Ollama({
  baseUrl: "http://localhost:11434", // Default value
  model: "mistral",
});

const query = "What is gravitational force?";
const prompt = PromptTemplate.fromTemplate(`You are a classifier trained on assigning label to input you receive. Please reduce the given prompt to one search word that can be used to research the topic on Wikipedia. Do only return the keyword.
Example: Prompt: How large is the Earth? Response: Size Earth
Example: Prompt: What is a pony? Response: Pony
Prompt:  {prompt}`);
const keyword = await prompt.pipe(llm).invoke({prompt:query });
console.info(keyword);
 const tool = new WikipediaQueryRun({
  topKResults: 3,
  maxDocContentLength: 4000,
});

const res = await tool.call(keyword);

const responsePrompt = PromptTemplate.fromTemplate(`You are a master instructor. Answer the prompt with the given context. Make sure to be friendly and encouraging. Only use the context provided and no further information. If the context isn't sufficient to answer the question, acknowledge this and ask the user to try again.
Prompt: {prompt}
Context: {context}`);
const answer = await responsePrompt.pipe(llm).invoke({
  prompt: query,
  context: res
})
console.log(answer);