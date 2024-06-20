# Twitter NLP 
The following code was developed to predict whether a twitter post informed us on a natural disaster. 
___
### Preprocessing the text
#### Removal of '@' from text
`spacy_cleaner` was used to create a text cleaning pipeline. The default methods within the library do not enable us to remove the twitter mentions (@).
Text Before Preprocessing: `"@aria_ahrary @TheTawniest The out of control wild fires in California even in the Northern part of the state. Very troubling."`
Text After Preprocessing: `"@aria_ahrary @thetawniest control wild fire california northern state troubling"`

To overcome this, custom **Evaluator** and **replacer** functions were created to remove the mentions. These custome functions were added inside the spacy_cleaner pipeline: 
Text After Custom Preprocessing: `"control wild fire california northern state troubling"`

___
### Running the Code 
#### Creating Virtual Environment
To create the virtual environment run the following code in your terminal, you can rename `env` to any name you want for your virtual environment.`python -m venv env`.

In the event the virtual environment has not yet been activate, you need to run the following command: `env\Scripts\activate.bat`. This might defer based on which machine you're using, I was using Visual Studio Code on a Windows and the command prompt as terminal. 

#### Install all the dependencies 
After creating the virtual environment, run the following command, this will download all the required libraries required to replicate the code. `python pip install -r requirements.txt`

#### Executing main.py
To run the main file run the following command within your terminal `python main.py`.

___
### Comments and Contribution 
This a project for Kaggle, any comment or contribution are welcome.