const exp = require('express');
const mult = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const cors = require('cors');

const app = exp()
app.use(cors());
const port = 3001;

const up = mult({storage: mult.memoryStorage()});

app.post('/api/scanner', up.single('image'), async (req, res) => {
    if (!req.file){
        return res.status(400).json({error: 'no image found'});
    }

    try{
        console.log('\n image successfully received: ${req.file.originalname} Sending it to the Meowth Guard AI...');

        const form = new FormData();
        form.append('file', req.file.buffer, req.file.originalname);

        const respon = await axios.post('http://localhost:5000/api/scan', form, {
            headers: {
                ...form.getHeaders()
            }
        });

        console.log("AI's response:", respon.data);

        res.json({
            success: true,
            ai_analysis:respon.data
        });

    } catch (error) {
        console.error("There was an issue getting in touch with the AI:", error.message);
        res.status(500).json({error: "Couldn't reach the AI"});
    }
});

app.listen(port, () => {
    console.log(`The AI is awaiting resquests on http://localhost:${port}`);
});