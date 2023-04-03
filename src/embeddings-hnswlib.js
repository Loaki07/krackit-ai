import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RetrievalQAChain } from "langchain/chains";
import { CSVLoader } from "langchain/document_loaders";
import { OpenAI } from "langchain/llms";


const loadDataToHNSWLib = async () => {
    try {

        const loader = new CSVLoader(
            "./assets/leetcode_dataset.csv",
        );

        const docs = await loader.load();

        // Load the docs into the vector store
        const vectorStore = await makeEmbeddedRequestHNSWLib(docs);

        // Save the vector store to a directory
        const directory = "./hnswlib-db";
        await vectorStore.save(directory);


        console.log(`load success`);
        return "load success";
    } catch (error) {
        console.error(`Error loading data to hnswlib: ${error}`);
    }
}

const queryHNSWLibDb = async (question) => {

    try {
        const model = new OpenAI({ modelName: "gpt-3.5-turbo" });

        const directory = "./hnswlib-db";

        // Load the vector store from the same directory
        const loadedVectorStore = await HNSWLib.load(
            directory,
            new OpenAIEmbeddings()
        );

        const chain = RetrievalQAChain.fromLLM(model, loadedVectorStore.asRetriever());
        const res = await chain.call({ query: question });
        console.log({ res });

        return res;

    } catch (error) {
        console.error(error);
    }
}

const backoffHNSWLib = (attempt) => {
    const waitTime = Math.pow(2, attempt) * 1000; // exponential backoff
    return new Promise(resolve => setTimeout(resolve, waitTime));
}

const makeEmbeddedRequestHNSWLib = async (texts) => {
    let attempt = 0;
    while (true) {
        try {
            console.log(`Backoff Request-HNSWLib Trial: ${attempt}`);
            const vectorStore = await HNSWLib.fromDocuments(texts, new OpenAIEmbeddings());
            return vectorStore;
        } catch (error) {
            console.log(error);
            if (error?.response?.status === 429) { // rate limit error
                attempt += 1;
                await backoffHNSWLib(attempt);
            } else {
                throw error;
            }
        }
    }
}

export { loadDataToHNSWLib, queryHNSWLibDb };