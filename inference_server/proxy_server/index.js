const express = require('express');
const app = express();

const port = 8383;
const bodyParser = require('body-parser');
const axios = require('axios')
var cors = require('cors');
app.use(express.json({
    limit : "100mb"
}));
app.use(express.urlencoded({
    limit:"100mb",
    extended: false
}));
출처: https://spiralmoon.tistory.com/entry/Nodejs-PayloadTooLargeError-request-entity-too-large [Spiral Moon's programming blog:티스토리]
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
app.use(cors());

app.post('/vqa', async (req, res) => {
    const { base64image, question } = req.body
    try {
        const result = await axios.post('http://127.0.0.1:8080/v1/models/vqa-model:predict', { base64image, question });
        res.send(result.data);
    } catch (err){
        console.log(err)
        res.status(500).send(err)
    }
});

app.listen(port, ()=>{
    console.log(`Express server has started on port ${port}`);
});
