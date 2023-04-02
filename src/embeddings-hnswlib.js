import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
import { CSVLoader } from "langchain/document_loaders";
import { OpenAI } from "langchain/llms";


const loadDataToHNSWLib = async () => {
    try {
        const llmA = new OpenAI({});
        const chainA = loadQAStuffChain(llmA);

        const loader = new CSVLoader(
            "./assets/leetcode_sample.csv",
        );

        const docs = await loader.load();

        // Load the docs into the vector store
        const vectorStore = await makeEmbeddedRequestHNSWLib(docs);

        // Save the vector store to a directory
        const directory = "./hnswlib-db";
        await vectorStore.save(directory);

        // Load the vector store from the same directory
        const loadedVectorStore = await HNSWLib.load(
            directory,
            new OpenAIEmbeddings()
        );

        const resA = await chainA.call({
            input_documents: loadedVectorStore,
            question: "Lists Question Asked by Amazon.",
        });
        console.log({ resA });

        return resA;
    } catch (error) {
        console.error(`Error loading data to hnswlib: ${error}`);
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
            console.log(error.response.status);
            if (error.response.status === 429) { // rate limit error
                attempt += 1;
                await backoffHNSWLib(attempt);
            } else {
                throw error;
            }
        }
    }
}

export default loadDataToHNSWLib;