# TACIT API

### Information
* This repository contains the code for TACIT api.
* DistillBERT and BERT models are present on the [SFSU Box](https://sfsu.app.box.com/folder/131399945083) folder. 
* Copy the contents of the <strong style="color:red;">tacit-backend</strong> to your machine.
* Download the distillbert model from the SFSU Box folder and place it under distillbert_model directory.
* To install the python packages and start the api run
```bash
cd  tacit-backend
pip install -r requirements.txt
python app.py
```
* The api will be available on http://localhost:5000/

### API Hosting
* The API needs to be hosted to make it available for end users.
* Since there are numerous dependencies and the model size is large, we were not able to host on smaller platforms like Heroku or AWS serverless.
* The API runs fine on an EC2 instance.
* Need to start an EC2 instance which has enough memory ~4GB.
* Copy the contents of the directory <strong style="color:red;">tacit-backend</strong> to EC2 instance, install the dependencies and then run the app. 
* EC2 instance has a public endpoint that can be accessed by the frontend.

### Improvements
* Check for methods to reduce the size of the model and dependencies so that it can be hosted easily.
* Training distillbert without the **ktrain** should help cut down the dependencies and reduce the model size. Training [Example](https://towardsdatascience.com/fine-tuning-hugging-face-model-with-custom-dataset-82b8092f5333)
