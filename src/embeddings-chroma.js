import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
import { CSVLoader } from "langchain/document_loaders";
import { OpenAI } from "langchain/llms";

const loadDataToChroma = async () => {
    try {
        const llmA = new OpenAI({});
        const chainA = loadQAStuffChain(llmA);

        const loader = new CSVLoader(
            "./assets/leetcode_sample.csv",
        );

        const docs = await loader.load();

        const vectorStore = await Chroma.fromDocuments(docs, new OpenAIEmbeddings(), {
            collectionName: "leetcode",
        });

        const resA = await chainA.call({
            input_documents: docs,
            question: "Lists Question Asked by Amazon.",
        });
        console.log({ resA });
        return resA;
    } catch (error) {
        console.error(`Error loading data to chroma: ${error}`);

    }
}

export default loadDataToChroma;