# Online Dojo Web Trainer

### Problem Statement (1-2 sentences)
Can I create a predictive model that accurately categorizes jabs in real time?
       
### Executive Summary (2-3 paragraphs)

I collected 1000+ picures of correct and incorrect jabs from my web cam and people at my muay thai gym. After carefully rating each jab as correct or incorrect i edited the images to add on the movenet lightning pose estimator. After adding the pose estimation skeleton I was ready to train the model based on what I had collected. It took a while to train but after hyper tuning the parameters I ended up with a 97% accuracy rate.

The amazing part is that this model is extremely accurate and will not accept even the smallest errors in technique. Feel free to try it out! 

To Try This Out Simply Clone the repository install the dependencies and run 'streamlit run Untitled.py' from the directory.

    
### File Directory/table of contents
./online_dojo/...
    ./jab - Trained Tensorflow CNN Model
    ./model.tflite - TFlite Movenet Lightning
    ./packages.txt - list of required packages to be downloaded
    ./requirements.txt - list of required libraries to be downloaded to venv
    ./README.md - Executive summary of the analysis
    


