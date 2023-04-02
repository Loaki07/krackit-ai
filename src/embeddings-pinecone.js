import { OpenAIEmbeddings } from "langchain/embeddings";
import { PineconeClient } from "@pinecone-database/pinecone";
import { OpenAI } from "langchain/llms";
import * as fs from "fs";
import { CSVLoader } from "langchain/document_loaders";
import { PineconeStore } from 'langchain/vectorstores';
import { VectorDBQAChain } from "langchain/chains";


const initModel = () => {
    // initialize LLM to answer the question
    return new OpenAI();
}

const initEmbeddings = () => {
    /*
    create embeddings to extract text from document and send to openAI for embeddings then
    add vectors to pinecone for storage
    */
    const embeddings = new OpenAIEmbeddings({
        maxConcurrency: 5,
        maxRetries: 2,
    });
    return embeddings;
}

const initPineCone = async () => {
    try {
        const pinecone = new PineconeClient;
        //initialize the vectorstore to store embeddings
        await pinecone.init({
            environment: `${process.env.PINECONE_ENVIRONMENT}`,
            apiKey: `${process.env.PINECONE_API_KEY}`,
        });
        return pinecone;
    } catch (error) {
        console.error(`Pinecone init error ${error}`);
    }
}

const addDocuments = async (documents) => {
    const texts = documents.map(({ pageContent }) => pageContent);
    const embedded = await makeEmbeddedRequest(texts);
    return addVectors(embedded, documents);
}

const addVectors = async (vectors, documents) => {
    //create the pinecone upsert object per vector
    const upsertRequest = {
        vectors: vectors.map((values, idx) => ({
            id: `${idx}`,
            metadata: {
                ...documents[idx].metadata,
                text: documents[idx].pageContent,
            },
            values,
        })),
        namespace: "leetcode",
    };

    fs.writeFileSync("./assets/leetcode_dataset.json", JSON.stringify(upsertRequest));

    await index.upsert({
        upsertRequest,
    });
}

const loadData = async () => {
    try {

        const pinecone = await initPineCone();

        const model = initModel();

        const loader = new CSVLoader(
            "./assets/leetcode_sample.csv",
        );

        const docs = await loader.load();

        //retrieve API operations for index created in pinecone dashboard
        const index = pinecone.Index("krackit-ai");


        await addDocuments(docs);

        let embeddings = initEmbeddings();

        // const docres = await PineconeStore.fromDocuments(docs, embeddings, {
        //     pineconeIndex: index,
        //     namespace: "leetcode",
        //     textKey: 'text',
        // });

        /*
        check your pinecone index dashboard to verify insertion
        run the code below to check your indexstats. You should see the new "namespace"
        */
        const indexData = await index.describeIndexStats({
            describeIndexStatsRequest: {},
        });
        console.log("indexData", indexData);

        const vectorStore = await PineconeStore.fromExistingIndex(
            new OpenAIEmbeddings(),
            { pineconeIndex: index }
        );

        const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
            k: 1,
            returnSourceDocuments: true,
        });
        const response = await chain.call({ query: "Amazon?" });
        console.log(response);
        return response;
    } catch (error) {
        console.error(`Data Loading error ${error}`);
    }
}

const backoff = (attempt) => {
    const waitTime = Math.pow(2, attempt) * 1000; // exponential backoff
    return new Promise(resolve => setTimeout(resolve, waitTime));
}

const makeEmbeddedRequest = async (texts) => {
    let attempt = 0;
    while (true) {
        try {
            console.log(`Backoff Request Trial: ${attempt}`);
            const embeddings = initEmbeddings();

            let embedded = await embeddings.embedDocuments(texts);
            return embedded;
        } catch (error) {
            console.log(error.response.status);
            if (error.response.status === 429) { // rate limit error
                attempt += 1;
                await backoff(attempt);
            } else {
                throw error;
            }
        }
    }
}

export default loadData;