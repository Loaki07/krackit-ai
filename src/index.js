import express, { response } from 'express'
import dotenv from 'dotenv'
import morgan from 'morgan'
import cors from 'cors'
import { loadDataToHNSWLib, queryHNSWLibDb } from './embeddings-hnswlib.js'
import loadDataToChroma from './embeddings-chroma.js'
import loadData from './embeddings-pinecone.js'
dotenv.config()

const app = express()

app.use(cors())
app.use(express.json())

if (process.env.NODE_ENV === 'development') {
    app.use(morgan('dev'))
}


app.get('/', async (req, res) => {
    // let response = await loadDataToHNSWLib();
    res.status(200).send({
        success: true,
        message: 'krackit-ai api backend!',
    });
})

app.get('/ask/:query', async (req, res) => {
    let response = await queryHNSWLibDb(req.params.query);
    res.status(200).send({
        success: true,
        message: 'AI Response',
        data: response,
    });
})

const PORT = process.env.PORT || 9797

app.listen(
    PORT,
    console.log(
        `Server running is ${process.env.NODE_ENV} mode on port ${PORT}`
    )
)